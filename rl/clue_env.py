import numpy as np
import logging
from ClueBasics.GameRules import GameRules
from agents.HeuristicsBot import HeuristicsBot
from agents.EliminationBot import EliminationBot
from agents.TriggerHappyBot import TriggerHappyBot


class ClueEnv:
    def __init__(self, num_players=3, rl_player_index=0, accusation_policy=None, reward_shaping=None, verbose=False):
        """
        accusation_policy: callable(state, action, env) -> bool
            Override when to accuse. Bypasses the accusation timing head entirely.
        reward_shaping: callable(prev_state, state, action, base_reward, env) -> float
            Applied on non-terminal steps only. Terminal rewards are never modified.
        verbose: bool
            If True, print game actions and deductions.
        """
        self.num_players      = num_players
        self.rl_player_index  = rl_player_index
        self.accusation_policy = accusation_policy
        self.reward_shaping   = reward_shaping
        self.verbose          = verbose
        self.action_space     = self._build_action_space()

        self.agent = None
        self.done  = True

        # Initialise episode-scoped tracking attributes so they exist before
        # reset() is called (e.g. compute_state_dim() is safe to call first).
        self._last_reveal     = []
        self._last_accusation = None

    # -------------------------------------------------------------------------
    # Action space

    def _build_action_space(self):
        actions = []
        for s in GameRules.SUSPECTS:
            for w in GameRules.WEAPONS:
                for r in GameRules.ROOMS:
                    actions.append((s, w, r))
        return actions

    # -------------------------------------------------------------------------
    # Reset

    def reset(self):
        self.game    = GameRules(players=[])
        self.game.verbose = self.verbose
        self.players = []
        self.done    = False
        self._last_reveal     = []
        self._last_accusation = None

        opponent_types = [
            (EliminationBot,  "EliminationBot"),
            (TriggerHappyBot, "TriggerHappyBot"),
            (HeuristicsBot,   "HeuristicsBot"),
        ]

        for i in range(self.num_players):
            if i == self.rl_player_index:
                # EliminationBot provides direct-observation belief tracking.
                # Its playTurn() is never called — DQN drives all decisions.
                # Using EliminationBot (not HeuristicsBot) is deliberate: the
                # belief matrix is grounded in direct observations only, giving
                # the suggestion head genuine room to discover why high-entropy,
                # high-probability suggestions are good rather than inheriting
                # that strategy from HeuristicsBot and slowly corrupting it.
                self.players.append(EliminationBot("RLAgent", self.game, "bot"))
            else:
                idx = (i if i < self.rl_player_index else i - 1) % len(opponent_types)
                bot_class, bot_name = opponent_types[idx]
                self.players.append(bot_class(f"{bot_name}_{i}", self.game, "bot"))

        for p in self.players:
            p.setOpponents([op for op in self.players if op != p])

        self.game.players = self.players
        self.game.dealCards()   # calls initialCrossOff() on every player

        self._log_game_start()
        return self.get_state()

    def set_agent(self, agent):
        """Call after constructing the DQNAgent so the env can use all three heads."""
        self.agent = agent

    def compute_state_dim(self):
        """
        State vector size (must match get_state() exactly):
          hand              : action_space_size
          belief_matrix     : (num_players + 1) * total_cards
          suggestion_history: N * (action_space_size + num_players + 2)
          accusation_context: 1 + (num_players - 1) + 3
                               turn_norm | per-opp unanswered | max sol-probs
        """
        action_size    = len(self.action_space)
        num_owners     = self.num_players + 1
        total_cards    = GameRules.totalCards
        N              = 5
        entry_size     = action_size + self.num_players + 2
        accusation_ctx = 1 + (self.num_players - 1) + 3
        return action_size + (num_owners * total_cards) + (N * entry_size) + accusation_ctx

    def _log_game_start(self):
        msg = "NEW GAME - hands: " + " ".join(
            f"[{p.name}: {', '.join(c.name for c in p.cards)}]"
            for p in self.players
        )
        logging.info(msg)

    # -------------------------------------------------------------------------
    # Step

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("step() called on a finished episode — call reset() first.")

        action     = self.action_space[action_idx]
        rl_player  = self.players[self.rl_player_index]
        suspect    = self._find_card_by_name(action[0], self.game.suspectCards)
        weapon     = self._find_card_by_name(action[1], self.game.weaponCards)
        room       = self._find_card_by_name(action[2], self.game.roomCards)
        prev_state = self.get_state()

        # Capture state now for the reveal chooser closure; it will be used
        # when opponents ask the RL agent to show a card this step.
        rl_reveal_chooser = self._make_reveal_chooser(prev_state)

        # RL agent makes its suggestion
        responder, shown_card = self.game.makeSuggestion(
            rl_player, suspect, weapon, room
        )

        # --- Belief update (EliminationBot level: direct observation only) ---
        # If someone showed a card to the RL agent, record the certainty and
        # renormalise that category's solution probabilities.  No Bayesian
        # smearing, no skipped-player inference — only what was directly seen.
        if shown_card is not None:
            rl_player.crossOff(responder, shown_card)

        if self.verbose:
            if shown_card is not None:
                print(f"{responder.name} showed a card — {shown_card.name}.")
            elif responder is None:
                print(f"No one could disprove {rl_player.name}'s suggestion.")
            print(f"  Possible: {[c.name for c in rl_player.possibleSuspects]} | "
                  f"{[c.name for c in rl_player.possibleWeapons]} | "
                  f"{[c.name for c in rl_player.possibleRooms]}")

        reward = 0.0
        done   = False
        info   = {}

        # --- Accusation timing decision ---
        # State is computed AFTER the suggestion and belief update so the
        # accusation head sees the most current information.
        accusation_state = self.get_state()

        if self.accusation_policy:
            # External policy override (e.g. evaluation or ablation)
            should_accuse = self.accusation_policy(accusation_state, action, self)

        elif self.agent is not None:
            # Accusation timing head with two-gate confidence check.
            # Gate 1 (min_confidence=0.30): max solution prob per category must
            #   exceed a floor before the head is even consulted.
            #   With EliminationBot backing these probs rise monotonically as
            #   cards are crossed off (1/6 → 1/5 → … → 1.0 for suspects),
            #   so the gate opens naturally at a reasonable point.
            # Gate 2: conservative exploration — epsilon fires → always wait.
            best_s = max(
                (rl_player.getProbability("Solution", c) for c in rl_player.possibleSuspects),
                default=0.0,
            )
            best_w = max(
                (rl_player.getProbability("Solution", c) for c in rl_player.possibleWeapons),
                default=0.0,
            )
            best_r = max(
                (rl_player.getProbability("Solution", c) for c in rl_player.possibleRooms),
                default=0.0,
            )
            should_accuse = self.agent.select_accusation(
                accusation_state, best_s, best_w, best_r
            )

        else:
            # Fallback when no agent is attached (e.g. scripted evaluation)
            should_accuse = self._default_accusation_policy(rl_player)

        # Record the accusation decision for the training loop
        self._last_accusation = {
            "state":  accusation_state,
            "action": int(should_accuse),
        }

        if should_accuse:
            # Pick the highest-probability candidate in each category.
            # When the agent is fully certain (only 1 possible left), this
            # trivially selects the right card.  When accusing early with
            # partial knowledge it picks the best guess in each category.
            acc_suspect = max(
                rl_player.possibleSuspects,
                key=lambda c: rl_player.getProbability("Solution", c),
            )
            acc_weapon = max(
                rl_player.possibleWeapons,
                key=lambda c: rl_player.getProbability("Solution", c),
            )
            acc_room = max(
                rl_player.possibleRooms,
                key=lambda c: rl_player.getProbability("Solution", c),
            )
            if rl_player.makeAccusation(acc_suspect, acc_weapon, acc_room):
                reward = 1.0
                done   = True
                info["result"] = "win"
            else:
                reward = -1.0
                done   = True
                info["result"] = "wrong_accusation"

        # --- Opponent turns ---
        if not done:
            next_i = (self.rl_player_index + 1) % self.num_players
            while next_i != self.rl_player_index:
                bot = self.players[next_i]
                if bot.inGame:
                    winner = self._run_bot_turn(bot, rl_reveal_chooser)
                    if winner:
                        reward = -1.0
                        done   = True
                        info["result"] = "bot_won"
                        break
                next_i = (next_i + 1) % self.num_players
            # EliminationBot-backed RL agent: no processNewSuggestions() call.
            # Opponent suggestions are not used to update the agent's belief
            # matrix — only direct observations (cards shown to the RL agent
            # in response to its own suggestions) count.

        # --- Reward shaping (non-terminal only) ---
        if not done and self.reward_shaping:
            reward = self.reward_shaping(prev_state, self.get_state(), action, reward, self)

        self.done = done
        return self.get_state(), reward, done, info

    def _make_reveal_chooser(self, state_at_decision_time):
        agent = self.agent

        def chooser(matching_cards):
            if agent is None or len(matching_cards) == 1:
                import random
                return random.choice(matching_cards)
            chosen = agent.select_reveal(
                state_at_decision_time,
                matching_cards,
                self.game.cards,
            )
            self._last_reveal.append({
                "state":     state_at_decision_time,
                "card":      chosen,
                "all_cards": self.game.cards,
            })
            return chosen

        self._last_reveal = []
        return chooser

    def _run_bot_turn(self, bot, rl_reveal_chooser):
        """
        Run a single bot turn, monkey-patching game.makeSuggestion so the
        RL agent's DQN reveal head controls which of its own cards it shows
        when opponents ask it to respond.
        """
        original_make_suggestion = self.game.makeSuggestion

        def patched_make_suggestion(player, perp, weapon, room, reveal_chooser=None):
            return original_make_suggestion(
                player, perp, weapon, room,
                reveal_chooser=rl_reveal_chooser,
            )

        self.game.makeSuggestion = patched_make_suggestion
        try:
            winner = bot.playTurn()
        finally:
            self.game.makeSuggestion = original_make_suggestion

        return winner

    def _default_accusation_policy(self, player):
        """
        Fallback used when no agent is attached.  Only accuses when fully certain:
        one possible card per category with solution probability ≥ 0.99.
        """
        if not (len(player.possibleSuspects) == 1 and
                len(player.possibleWeapons)  == 1 and
                len(player.possibleRooms)    == 1):
            return False
        s = player.possibleSuspects[0]
        w = player.possibleWeapons[0]
        r = player.possibleRooms[0]
        return (player.getProbability("Solution", s) >= 0.99 and
                player.getProbability("Solution", w) >= 0.99 and
                player.getProbability("Solution", r) >= 0.99)

    # -------------------------------------------------------------------------
    # State

    def get_state(self):
        """
        State vector layout:
          [hand | belief_matrix | suggestion_history | accusation_context]

        hand (action_space_size):
            Binary; 1.0 if the agent holds any card in that (s,w,r) tuple.

        belief_matrix (num_owners * num_cards):
            EliminationBot-level. Certainty values (0 or 1) for directly
            observed cards; uniform priors for the rest, normalised within
            each category as cards are crossed off.

        suggestion_history (N=5 × entry_size):
            Per entry: one-hot action, one-hot suggester, was_refuted, is_rl.

        accusation_context (1 + (num_players-1) + 3 floats):
            turn_norm         — how deep into the game we are (0→1).
            opp_unanswered[i] — fraction of opponent i's suggestions that went
                                unrefuted; proxy for how close they are to solving.
            max_sol_probs     — best solution probability in each category
                                (suspect, weapon, room); rises toward 1.0 as
                                the agent crosses off cards.  Direct input to the
                                accusation timing head's confidence gate.
        """
        player      = self.players[self.rl_player_index]
        action_size = len(self.action_space)
        num_players = self.num_players

        # Hand encoding
        hand = np.zeros(action_size, dtype=np.float32)
        for card in player.cards:
            for i, (s, w, r) in enumerate(self.action_space):
                if card.name in (s, w, r):
                    hand[i] = 1.0

        # Belief matrix
        belief_matrix = np.array(
            [player.ownersAndCards.get(owner, {}).get(card, 0.0)
             for owner in player.owners
             for card in self.game.cards],
            dtype=np.float32,
        )

        # Suggestion history (last N=5)
        N          = 5
        entry_size = action_size + num_players + 2
        history    = np.zeros((N, entry_size), dtype=np.float32)
        public_log = self.game.getPublicSuggestionLog(player)
        p_idx      = {p: i for i, p in enumerate(self.players)}

        for i, rec in enumerate(public_log[-N:]):
            names = tuple(c.name for c in rec["suggestion"])
            try:
                history[i, self.action_space.index(names)] = 1.0
            except ValueError:
                pass
            suggester = rec.get("suggester")
            if suggester in p_idx:
                history[i, action_size + p_idx[suggester]] = 1.0
            history[i, action_size + num_players]     = float(rec.get("responder") is not None)
            history[i, action_size + num_players + 1] = float(suggester is player)

        # Accusation context
        # 1. Normalised turn counter (proxy for game depth)
        turn_norm = np.float32(min(self.game.turn / 200.0, 1.0))

        # 2. Per-opponent unanswered-suggestion fraction.
        #    "Unanswered" = no one could refute → strong signal that the opponent
        #    has narrowed those three cards and may be close to accusing.
        opp_order = [p for i, p in enumerate(self.players) if i != self.rl_player_index]
        opp_idx   = {p: i for i, p in enumerate(opp_order)}
        opp_unanswered = np.zeros(num_players - 1, dtype=np.float32)
        for rec in public_log:
            if rec["suggester"] is not player and rec["responder"] is None:
                oi = opp_idx.get(rec["suggester"])
                if oi is not None:
                    opp_unanswered[oi] += 1.0
        total_suggestions = max(len(public_log), 1)
        opp_unanswered /= total_suggestions

        # 3. Best solution probability per category.
        #    With EliminationBot normalisation, these rise monotonically as
        #    cards are ruled out.  When all three hit 1.0 the answer is known.
        max_s_prob = max(
            (player.getProbability("Solution", c) for c in player.possibleSuspects),
            default=0.0,
        )
        max_w_prob = max(
            (player.getProbability("Solution", c) for c in player.possibleWeapons),
            default=0.0,
        )
        max_r_prob = max(
            (player.getProbability("Solution", c) for c in player.possibleRooms),
            default=0.0,
        )
        max_sol_probs = np.array([max_s_prob, max_w_prob, max_r_prob], dtype=np.float32)

        accusation_ctx = np.concatenate([[turn_norm], opp_unanswered, max_sol_probs])

        return np.concatenate([hand, belief_matrix, history.flatten(), accusation_ctx])

    def get_state_dim(self):
        return self.get_state().shape[0]

    # -------------------------------------------------------------------------
    # Legal actions

    def get_legal_actions(self):
        """
        Filter out actions where the agent holds all three cards.
        Suggesting own cards for bluffing is allowed as long as at least
        one card is a potential solution card.
        """
        player     = self.players[self.rl_player_index]
        hand_names = {c.name for c in player.cards}
        return [
            i for i, (s, w, r) in enumerate(self.action_space)
            if not (s in hand_names and w in hand_names and r in hand_names)
        ]

    # -------------------------------------------------------------------------
    # Helpers

    def get_suggestion_history(self):
        return self.game.getPublicSuggestionLog(self.players[self.rl_player_index])

    def _find_card_by_name(self, name, card_dict):
        for card in card_dict.values():
            if card.name == name:
                return card
        raise ValueError(f"Card '{name}' not found")
