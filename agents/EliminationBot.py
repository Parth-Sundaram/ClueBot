from ClueBasics.Player import Player
import random


class EliminationBot(Player):
    """
    Direct-observation belief tracker with random suggestion strategy.

    This is the RL agent's backing player class. It provides:
      - A belief matrix grounded solely in what the agent has directly observed
        (cards dealt to it at startup; cards shown to it during suggestions).
      - Solution-probability normalisation after each direct observation so the
        belief matrix stays a valid probability distribution as cards are ruled out.
      - No Bayesian smearing, no skipped-player inference, no cascading deduction.

    The DQN suggestion head has genuine room to discover strategy from this
    sparse-but-honest belief state, rather than inheriting near-optimal heuristics
    from HeuristicsBot and then slowly corrupting them as epsilon decays.
    """

    # ------------------------------------------------------------------
    # Setup

    def initialCrossOff(self):
        """
        Build the belief matrix then mark every card in hand as definitely
        held by self — so they're excluded from possible solution candidates
        and the Solution column gets renormalised accordingly.
        """
        self.createBeliefMatrix()
        for card in self.cards:
            self.crossOff(self, card)

    # ------------------------------------------------------------------
    # Core belief update

    def crossOff(self, owner, card):
        """
        Record a direct, certain observation: *owner* has *card*.

        Three effects:
          1. Remove card from the relevant possible-solution category list.
          2. Set P(owner, card) = 1; P(all other owners, card) = 0.
          3. Renormalise the Solution column for the remaining possible cards
             in that category so solution probs still sum to 1.

        Effect (3) means the accusation head sees rising max-solution-probs
        as cards get ruled out (e.g. 1/6 → 1/5 → 1/4 → … → 1.0 for suspects),
        giving it a clean, monotonically informative accusation signal without
        any Bayesian machinery.
        """
        # 1. Remove from possible-solution lists
        if card.getType() == 'Suspect':
            if card in self.possibleSuspects:
                self.possibleSuspects.remove(card)
        elif card.getType() == 'Weapon':
            if card in self.possibleWeapons:
                self.possibleWeapons.remove(card)
        else:
            if card in self.possibleRooms:
                self.possibleRooms.remove(card)

        # 2. Assert certainty in belief matrix
        for o in self.owners:
            self.setProbability(o, card, 1.0 if o == owner else 0.0)

        # 3. Renormalise Solution column for remaining possible cards
        if card.getType() == 'Suspect':
            remaining = self.possibleSuspects
        elif card.getType() == 'Weapon':
            remaining = self.possibleWeapons
        else:
            remaining = self.possibleRooms

        if remaining:
            total = sum(self.getProbability("Solution", c) for c in remaining)
            if total > 0:
                for c in remaining:
                    self.setProbability(
                        "Solution", c,
                        self.getProbability("Solution", c) / total,
                    )

    # ------------------------------------------------------------------
    # Strategy (random — the DQN learns on top of this)

    def chooseSuggestion(self):
        suspect = random.choice(self.possibleSuspects)
        weapon  = random.choice(self.possibleWeapons)
        room    = random.choice(self.possibleRooms)
        return suspect, weapon, room

    # ------------------------------------------------------------------
    # Turn (used when EliminationBot is an *opponent*, not the RL agent)

    def playTurn(self):
        if not self.inGame:
            return

        perp, weapon, room = self.chooseSuggestion()
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)

        if owner is not None:
            if self.game.verbose:
                print(f"{owner.name} showed a card.")
            self.crossOff(owner, card)

        # Accuse only when fully certain
        if (len(self.possibleSuspects) == 1 and
                len(self.possibleWeapons)  == 1 and
                len(self.possibleRooms)    == 1):
            if self.makeAccusation(
                    self.possibleSuspects[0],
                    self.possibleWeapons[0],
                    self.possibleRooms[0]):
                if self.game.verbose:
                    print(f"{self.name} WINS! The solution was correct.")
                return self.name
            else:
                if self.game.verbose:
                    print(f"{self.name} made a wrong accusation and is out.")
