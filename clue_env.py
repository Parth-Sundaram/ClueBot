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
            Override when to accuse. Default: only when solution is fully known.
        reward_shaping: callable(prev_state, state, action, base_reward, env) -> float
            Applied on non-terminal steps only. Terminal rewards are never modified.
        verbose: bool
            If True, print game actions and deductions. If False, minimal output.
        """
        self.num_players      = num_players
        self.rl_player_index  = rl_player_index
        self.accusation_policy = accusation_policy
        self.reward_shaping   = reward_shaping
        self.verbose          = verbose
        self.action_space     = self._build_action_space()

        # agent reference is set via set_agent() before training starts
        self.agent = None

        # Mark as needing reset — caller will call set_agent() then reset()
        self.done = True

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

        opponent_types = [
            (EliminationBot,  "EliminationBot"),
            (TriggerHappyBot, "TriggerHappyBot"),
            (HeuristicsBot,   "HeuristicsBot"),
        ]

        for i in range(self.num_players):
            if i == self.rl_player_index:
                # HeuristicsBot provides belief matrix tracking.
                # Its playTurn() is never called — DQN drives all decisions.
                self.players.append(HeuristicsBot("RLAgent", self.game, "bot"))
            else:
                idx = (i if i < self.rl_player_index else i - 1) % len(opponent_types)
                bot_class, bot_name = opponent_types[idx]
                self.players.append(bot_class(f"{bot_name}_{i}", self.game, "bot"))

        for p in self.players:
            p.setOpponents([op for op in self.players if op != p])

        self.game.players = self.players
        # dealCards() calls initialCrossOff() on every player internally
        self.game.dealCards()

        self._log_game_start()
        return self.get_state()

    def set_agent(self, agent):
        """Call after constructing the DQNAgent so the env can use the reveal head."""
        self.agent = agent

    def compute_state_dim(self):
        """
        Compute state vector size without running a full game.
        hand: action_space_size
        belief: (num_players + 1) * total_cards   (+1 for Solution)
        history: N * (action_space_size + num_players + 2)
        """
        action_size = len(self.action_space)
        num_owners  = self.num_players + 1  # players + Solution
        total_cards = GameRules.totalCards
        N           = 5
        entry_size  = action_size + self.num_players + 2
        return action_size + (num_owners * total_cards) + (N * entry_size)

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

        # Build the reveal chooser for the RL agent.
        # This closure is passed down through makeSuggestion -> refuteSuggestion -> showCard.
        # It's only invoked if the RL agent ends up being the responder to its own suggestion
        # (which can't happen — a player never responds to themselves), OR more usefully,
        # when an opponent suggests and the RL agent has matching cards (handled in opponent loop).
        rl_reveal_chooser = self._make_reveal_chooser(prev_state)

        # RL agent makes its suggestion
        responder, shown_card = self.game.makeSuggestion(
            rl_player, suspect, weapon, room
            # No reveal_chooser needed here: the RL agent is the suggester,
            # so it will never be the one choosing what to reveal in this call.
        )



        # Update RL agent's belief from what was shown
        if shown_card is not None:
            if self.verbose:
                print(f"{responder.name} showed a card - {shown_card.name}.")
            rl_player.crossOff(responder, shown_card)
        elif responder is None:
            if self.verbose:
                print(f"No one could disprove {rl_player.name}'s suggestion.")

        # DEBUG: print belief matrix state if verbose
        if self.verbose:
            print(f"  Possible: {[c.name for c in rl_player.possibleSuspects]} | "
                  f"{[c.name for c in rl_player.possibleWeapons]} | "
                  f"{[c.name for c in rl_player.possibleRooms]}")

        reward = 0.0
        done   = False
        info   = {}

        # --- Accusation ---
        if self.accusation_policy:
            should_accuse = self.accusation_policy(self.get_state(), action, self)
        else:
            should_accuse = self._default_accusation_policy(rl_player)

        if should_accuse:
            if (len(rl_player.possibleSuspects) == 1 and len(rl_player.possibleWeapons) == 1 and len(rl_player.possibleRooms) == 1):
                acc_suspect = rl_player.possibleSuspects[0]
                acc_weapon  = rl_player.possibleWeapons[0]
                acc_room    = rl_player.possibleRooms[0]
            else:
                acc_suspect, acc_weapon, acc_room = suspect, weapon, room
            if rl_player.makeAccusation(acc_suspect, acc_weapon, acc_room):
                reward = 1.0
                done   = True
                info["result"] = "win"
            else:
                reward = -1.0
                done   = True
                info["result"] = "wrong_accusation"

        # --- Opponent turns ---
        # The reveal_chooser is passed to each opponent's makeSuggestion so that
        # if an opponent suggests cards the RL agent holds, the DQN reveal head picks
        # which card to show rather than the default random/heuristic.
        if not done:
            next_i = (self.rl_player_index + 1) % self.num_players
            while next_i != self.rl_player_index:
                bot = self.players[next_i]
                if bot.inGame:
                    # Inject reveal_chooser so RL agent controls its own reveals
                    winner = self._run_bot_turn(bot, rl_reveal_chooser)
                    if winner:
                        reward = -1.0
                        done   = True
                        info["result"] = "bot_won"
                        break
                next_i = (next_i + 1) % self.num_players

            if not done:
                rl_player.processNewSuggestions()

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
                self.game.cards
            )
            self._last_reveal.append({
                "state":     state_at_decision_time,
                "card":      chosen,
                "all_cards": self.game.cards,
            })
            return chosen

        self._last_reveal = []   # ← here, just before return
        return chooser

    def _run_bot_turn(self, bot, rl_reveal_chooser):
        """
        Run a single bot turn, injecting rl_reveal_chooser into the game so the
        RL agent controls its own card reveals when opponents ask it to respond.

        We monkey-patch game.makeSuggestion temporarily so the chooser is threaded
        through without modifying each bot's playTurn() signature.
        """
        original_make_suggestion = self.game.makeSuggestion

        def patched_make_suggestion(player, perp, weapon, room, reveal_chooser=None):
            return original_make_suggestion(
                player, perp, weapon, room,
                reveal_chooser=rl_reveal_chooser  # always inject, GameRules decides who uses it
    )

        self.game.makeSuggestion = patched_make_suggestion
        try:
            winner = bot.playTurn()
        finally:
            self.game.makeSuggestion = original_make_suggestion

        return winner

    def _default_accusation_policy(self, player):
        if not (len(player.possibleSuspects) == 1 and
                len(player.possibleWeapons)  == 1 and
                len(player.possibleRooms)    == 1):
            return False
        # Extra safety check: confirm solution probabilities are actually high
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
          hand_encoding      (action_space_size)
          belief_matrix      (num_owners * num_cards)
          suggestion_history (N * (action_space_size + num_players + 2))

        suggestion_history encodes per entry:
          - one-hot over action space   (what was suggested)
          - one-hot over players        (who suggested)
          - was_refuted flag            (0/1)
          - suggester_is_rl flag        (0/1)
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
            dtype=np.float32
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

        return np.concatenate([hand, belief_matrix, history.flatten()])

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
