import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# =============================================================================
# Network
# =============================================================================

class DQNetwork(nn.Module):
    """
    Shared-trunk network with three output heads.

    suggestion_head  — Q-values over all (suspect, weapon, room) combos.
                       Primary decision each turn. Gradients flow through trunk.

    reveal_head      — Q-values over MAX_HAND_SIZE card slots.
                       Decides which matching card to show an opponent.
                       Trunk gradients detached during reveal-only updates.

    accusation_head  — Q-values over {0=wait, 1=accuse}.
                       Learns *when* to accuse rather than relying on a hardcoded
                       certainty threshold. Novel contribution: the head sees
                       opponent-progress proxies (unanswered suggestion counts,
                       turn depth) alongside the full belief state, so it can
                       learn to gamble earlier if an opponent is close to solving.
                       Trunk gradients detached during accusation-only updates
                       so the accusation head reads shared features without
                       corrupting what the suggestion head learned.
    """
    MAX_HAND_SIZE = 7  # max cards in hand for a 3-player game

    def __init__(self, state_dim, suggestion_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.suggestion_head = nn.Linear(hidden_dim, suggestion_dim)
        self.reveal_head     = nn.Linear(hidden_dim, self.MAX_HAND_SIZE)
        self.accusation_head = nn.Linear(hidden_dim, 2)   # [Q_wait, Q_accuse]

    def forward(self, x):
        f = self.trunk(x)
        return self.suggestion_head(f), self.reveal_head(f), self.accusation_head(f)

    def suggestion_q(self, x):
        return self.suggestion_head(self.trunk(x))

    def reveal_q(self, x):
        return self.reveal_head(self.trunk(x))

    def accusation_q(self, x):
        return self.accusation_head(self.trunk(x))


# =============================================================================
# Agent
# =============================================================================

class DQNAgent:
    def __init__(
        self,
        state_dim,
        suggestion_dim,
        lr=1e-3,
        gamma=0.99,
        # --- suggestion exploration ---
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,          # per-episode
        # --- accusation exploration ---
        # Starts very high so the agent almost always waits early in training.
        # Conservative exploration (always wait when epsilon fires) means we never
        # make garbage accusations during the high-epsilon phase.
        accusation_epsilon_start=0.95,
        accusation_epsilon_end=0.05,
        accusation_epsilon_decay=0.998,  # decays slower than suggestion epsilon
        # --- buffers ---
        buffer_size=20000,
        reveal_buffer_size=5000,
        accusation_buffer_size=10000,
        batch_size=64,
        device=None,
    ):
        self.state_dim      = state_dim
        self.suggestion_dim = suggestion_dim
        self.gamma          = gamma
        self.epsilon                  = epsilon_start
        self.epsilon_end              = epsilon_end
        self.epsilon_decay            = epsilon_decay
        self.accusation_epsilon       = accusation_epsilon_start
        self.accusation_epsilon_end   = accusation_epsilon_end
        self.accusation_epsilon_decay = accusation_epsilon_decay
        self.batch_size     = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQNetwork(state_dim, suggestion_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, suggestion_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Three separate replay buffers — accusation choices are dense (one per step)
        # but the interesting signal (accuse decisions) is rare, so a mid-size buffer
        # keeps the wait/accuse ratio manageable.
        self.suggestion_memory  = deque(maxlen=buffer_size)
        self.reveal_memory      = deque(maxlen=reveal_buffer_size)
        self.accusation_memory  = deque(maxlen=accusation_buffer_size)

    # -------------------------------------------------------------------------
    # Suggestion head

    def select_suggestion(self, state, legal_actions=None):
        """
        Epsilon-greedy selection over the suggestion head.
        Illegal actions (agent holds all three cards) are masked to -inf.
        """
        if legal_actions is None:
            legal_actions = list(range(self.suggestion_dim))

        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net.suggestion_q(state_t).squeeze(0).cpu().numpy()

        masked = np.full(self.suggestion_dim, -np.inf)
        masked[legal_actions] = q[legal_actions]
        return int(np.argmax(masked))

    def store_suggestion(self, state, action, reward, next_state, done):
        self.suggestion_memory.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def update_suggestion(self):
        """Double DQN update on suggestion head. Gradients flow through trunk."""
        if len(self.suggestion_memory) < self.batch_size:
            return None

        batch = random.sample(self.suggestion_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.stack(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net.suggestion_q(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net.suggestion_q(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net.suggestion_q(next_states).gather(1, next_actions)
            target       = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    # -------------------------------------------------------------------------
    # Reveal head

    def select_reveal(self, state, matching_cards, all_cards):
        """
        Choose which card to reveal from matching_cards.
        Slots are assigned by position in all_cards mod MAX_HAND_SIZE.
        """
        max_slots    = DQNetwork.MAX_HAND_SIZE
        card_to_slot = {card: i % max_slots for i, card in enumerate(all_cards)}
        legal_slots  = [card_to_slot[c] for c in matching_cards]

        if len(matching_cards) == 1:
            return matching_cards[0]

        if np.random.rand() < self.epsilon:
            return random.choice(matching_cards)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net.reveal_q(state_t).squeeze(0).cpu().numpy()

        masked = np.full(max_slots, -np.inf)
        for slot in legal_slots:
            masked[slot] = q[slot]

        chosen_slot  = int(np.argmax(masked))
        slot_to_card = {s: c for s, c in zip(legal_slots, matching_cards)}
        return slot_to_card[chosen_slot]

    def store_reveal(self, state, chosen_card, all_cards, reward, next_state, done):
        max_slots    = DQNetwork.MAX_HAND_SIZE
        card_to_slot = {card: i % max_slots for i, card in enumerate(all_cards)}
        slot = card_to_slot[chosen_card]
        self.reveal_memory.append((
            np.array(state,      dtype=np.float32),
            int(slot),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def update_reveal(self):
        """
        Update reveal head. Trunk detached — reveal updates don't corrupt the
        shared features learned by the suggestion head.
        """
        min_batch = max(8, self.batch_size // 4)
        if len(self.reveal_memory) < min_batch:
            return None

        batch = random.sample(self.reveal_memory, min_batch)
        states, slots, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.stack(states)).to(self.device)
        slots       = torch.LongTensor(slots).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        trunk_features      = self.policy_net.trunk(states).detach()
        next_trunk_features = self.target_net.trunk(next_states).detach()

        q_values = self.policy_net.reveal_head(trunk_features).gather(1, slots)
        with torch.no_grad():
            next_q  = self.target_net.reveal_head(next_trunk_features).max(1, keepdim=True)[0]
            target  = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    # -------------------------------------------------------------------------
    # Accusation timing head

    def select_accusation(
        self,
        state,
        best_suspect_prob: float = 0.0,
        best_weapon_prob:  float = 0.0,
        best_room_prob:    float = 0.0,
        min_confidence:    float = 0.30,
    ) -> bool:
        """
        Return True if the agent should accuse this turn.

        Two-gate design before the Q-value is even consulted:

        Gate 1 — Confidence floor:
            If the best solution probability in any category is below
            min_confidence, always wait.  With EliminationBot backing,
            this rises monotonically as cards are eliminated
            (1/N_category → … → 1.0), so the gate naturally opens later
            in better-informed games.  At 0.30 it fires when roughly 2/3
            of a category has been crossed off — a reasonable minimum for
            considering an accusation.

        Gate 2 — Conservative exploration:
            When accusation_epsilon fires, always choose WAIT (not a random
            50/50 coin flip).  Random accusations devastate episodes and
            produce useless gradients early in training.  As epsilon decays
            the agent increasingly trusts its learned Q-values.

        Once both gates pass, accuse iff Q(accuse) > Q(wait).
        """
        # Gate 1 — need at least some confidence in every category
        if (best_suspect_prob < min_confidence or
                best_weapon_prob  < min_confidence or
                best_room_prob    < min_confidence):
            return False

        # Gate 2 — conservative exploration: wait, not random
        if np.random.rand() < self.accusation_epsilon:
            return False

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net.accusation_q(state_t).squeeze(0).cpu().numpy()
        return bool(q[1] > q[0])

    def store_accusation(self, state, action, reward, next_state, done):
        """
        action : 0 = wait, 1 = accuse
        reward : +1.0 correct accusation / win
                 -1.0 wrong accusation or bot wins
                  0.0 wait on a non-terminal step
        Storing 0 for non-terminal waits keeps the signal clean — the
        accusation head learns purely from discounted terminal outcomes.
        """
        self.accusation_memory.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def update_accusation(self):
        """
        Double DQN update on the accusation timing head.

        Trunk is detached so this update only trains the accusation head
        linear layer — shared features remain owned by the suggestion head.
        The accusation head therefore learns to read the trunk's feature
        representation rather than reshaping it.

        What the head can learn:
          - High max-solution-probs across all three categories → accuse soon.
          - Low turn count + high probs → don't rush, opponent isn't close yet.
          - Opponent has several unanswered suggestions (state encodes this) →
            raise urgency even at moderate confidence levels.
        """
        min_batch = max(8, self.batch_size // 4)
        if len(self.accusation_memory) < min_batch:
            return None

        batch = random.sample(self.accusation_memory, min_batch)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.stack(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Detach trunk: only the accusation head's weights are updated
        trunk_features      = self.policy_net.trunk(states).detach()
        next_trunk_features = self.target_net.trunk(next_states).detach()

        q_values = self.policy_net.accusation_head(trunk_features).gather(1, actions)
        with torch.no_grad():
            # Double DQN: policy picks next action, target evaluates it
            next_actions = self.policy_net.accusation_head(next_trunk_features).argmax(1, keepdim=True)
            next_q       = self.target_net.accusation_head(next_trunk_features).gather(1, next_actions)
            target       = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    # -------------------------------------------------------------------------
    # Shared utilities

    def decay_epsilon(self):
        """
        Call once per episode.
        Both epsilons decay independently so the accusation head can be
        kept in conservative mode longer than suggestion exploration.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )
        self.accusation_epsilon = max(
            self.accusation_epsilon_end,
            self.accusation_epsilon * self.accusation_epsilon_decay,
        )

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy_net':         self.policy_net.state_dict(),
            'target_net':         self.target_net.state_dict(),
            'optimizer':          self.optimizer.state_dict(),
            'epsilon':            self.epsilon,
            'accusation_epsilon': self.accusation_epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon            = ckpt['epsilon']
        # Backward-compatible: old checkpoints may not have accusation_epsilon
        self.accusation_epsilon = ckpt.get('accusation_epsilon', self.accusation_epsilon)
