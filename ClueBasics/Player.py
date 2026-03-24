from .GameRules import GameRules
import random
from .Card import Card
from abc import ABC, abstractmethod


class Player(ABC):

    def __init__(self, name, game, type):
        self.name = name
        self.game = game
        self.possibleSuspects = list(game.suspectCards.values())
        self.possibleWeapons = list(game.weaponCards.values())
        self.possibleRooms = list(game.roomCards.values())
        self.inGame = True
        self.cards = []
        self.numCards = 0
        self.type = type
        self.privateSuggestionLog = []
        self.owners = []

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return str(self)

    def getNumCards(self):
        return self.numCards

    def setOpponents(self, opponents):
        self.opponents = opponents
        self.players = self.opponents + [self]
        self.owners = self.opponents + [self] + ["Solution"]
        self.ownersAndCards = {owner: {} for owner in self.owners}

    def getProbability(self, owner, card):
        return self.ownersAndCards[owner][card]

    def setProbability(self, owner, card, prob):
        self.ownersAndCards[owner][card] = prob

    def createBeliefMatrix(self):
        self.players = self.opponents + [self]
        self.owners = self.opponents + [self] + ["Solution"]
        total_cards = self.game.totalCards

        for owner in self.owners:
            self.ownersAndCards[owner] = {}

        for player in self.players:
            num_cards = player.getNumCards()
            for card in self.game.cards:
                self.ownersAndCards[player][card] = num_cards / total_cards

        for card in self.game.suspectCards.values():
            self.ownersAndCards["Solution"][card] = 1 / len(self.game.SUSPECTS)

        for card in self.game.weaponCards.values():
            self.ownersAndCards["Solution"][card] = 1 / len(self.game.WEAPONS)

        for card in self.game.roomCards.values():
            self.ownersAndCards["Solution"][card] = 1 / len(self.game.ROOMS)

    def isDealt(self, card):
        self.cards.append(card)
        self.numCards += 1

    # -------------------------------------------------------------------------
    # Helpers

    def crossOffMulti(self, owner, cardList):
        for card in cardList:
            if card.getType() == 'Suspect':
                self.possibleSuspects.remove(card)
            elif card.getType() == 'Weapon':
                self.possibleWeapons.remove(card)
            else:
                self.possibleRooms.remove(card)

    def revealCards(self):
        print(f"{self.name} has cards: ")
        for i, card in enumerate(self.cards):
            print(f"{i + 1}. {card.name}")

    def crossOff(self, owner, card):
        if card.getType() == 'Suspect':
            if card in self.possibleSuspects:
                self.possibleSuspects.remove(card)
        elif card.getType() == 'Weapon':
            if card in self.possibleWeapons:
                self.possibleWeapons.remove(card)
        else:
            if card in self.possibleRooms:
                self.possibleRooms.remove(card)

    def hasACard(self, perp, weapon, room):
        for card in self.cards:
            if card == perp or card == weapon or card == room:
                return card
        return None

    def initialCrossOff(self):
        self.createBeliefMatrix()

    # -------------------------------------------------------------------------
    # Player mechanics

    def updateBeliefs(self):
        pass

    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)

    @abstractmethod
    def chooseSuggestion(self):
        pass

    def showCard(self, matching_cards, reveal_chooser=None):
        """
        Choose which matching card to reveal.

        reveal_chooser: optional callable(matching_cards) -> card
            If provided (injected by the RL environment for the DQN reveal head),
            delegates the decision to the caller.
            Otherwise falls back to this player's own strategy (random by default;
            subclasses can override for heuristic reveal logic).
        """
        if reveal_chooser is not None:
            return reveal_chooser(matching_cards)
        return self._choose_card_to_show(matching_cards)

    def _choose_card_to_show(self, matching_cards):
        """
        Default reveal strategy: random.
        Override in subclasses for heuristic strategies
        (e.g. always reveal the card most likely already known to the suggester).
        """
        return random.choice(matching_cards)

    def refuteSuggestion(self, suggestionCards, reveal_chooser=None):
        """
        Returns the card shown to the suggester, or None if the player
        has no matching cards.

        reveal_chooser is passed through to showCard so the RL agent's
        DQN reveal head can override the selection when this player is
        the RL agent.
        """
        matching_cards = [card for card in self.cards if card in suggestionCards]
        if not matching_cards:
            return None
        return self.showCard(matching_cards, reveal_chooser=reveal_chooser)

    def makeSuggestion(self, perp, weapon, room):
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        return owner, card

    @abstractmethod
    def playTurn(self):
        pass
