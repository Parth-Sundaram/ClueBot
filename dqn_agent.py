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
    Shared-trunk network with two output heads:

    suggestion_head: Q-values over all (suspect, weapon, room) combinations.
                     This is the primary decision — what to suggest each turn.

    reveal_head:     Q-values over MAX_HAND_SIZE card slots.
                     This is the secondary decision — which matching card to
                     show an opponent when forced to respond to their suggestion.

    Both heads share the same trunk. The suggestion head is trained with the
    full win/loss reward signal (gradients flow through trunk).
    The reveal head is trained with the same win/loss signal but trunk gradients
    are detached during reveal-only updates so the two heads don't destabilise
    each other. See DQNAgent.update_reveal().
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

    def forward(self, x):
        f = self.trunk(x)
        return self.suggestion_head(f), self.reveal_head(f)

    def suggestion_q(self, x):
        return self.suggestion_head(self.trunk(x))

    def reveal_q(self, x):
        return self.reveal_head(self.trunk(x))


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
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,        # per-episode, not per-step
        buffer_size=20000,
        reveal_buffer_size=5000,    # reveal choices are rarer, smaller buffer is fine
        batch_size=64,
        device=None
    ):
        self.state_dim      = state_dim
        self.suggestion_dim = suggestion_dim
        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQNetwork(state_dim, suggestion_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, suggestion_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Two separate replay buffers
        self.suggestion_memory = deque(maxlen=buffer_size)
        self.reveal_memory     = deque(maxlen=reveal_buffer_size)

    # -------------------------------------------------------------------------
    # Suggestion head

    def select_suggestion(self, state, legal_actions=None):
        """
        Epsilon-greedy selection over suggestion head.
        legal_actions: list of valid action indices — illegal ones are masked to -inf.
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
            float(done)
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
        This gives each card a stable slot index regardless of hand size.

        Returns the chosen Card object.
        """
        max_slots    = DQNetwork.MAX_HAND_SIZE
        card_to_slot = {card: i % max_slots for i, card in enumerate(all_cards)}
        legal_slots  = [card_to_slot[c] for c in matching_cards]

        # Trivial case — no choice to make
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
            float(done)
        ))

    def update_reveal(self):
        """
        Update reveal head using the same win/loss reward signal as suggestion.
        Trunk gradients are detached so reveal-only updates don't corrupt the
        shared features learned by the suggestion head.

        To experiment with a reveal-specific reward (e.g. penalise revealing a
        card the opponent didn't know you held), replace `rewards` with a shaped
        tensor before the target computation below.
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

        # Detach trunk: reveal updates don't touch shared features
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
    # Shared utilities

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
