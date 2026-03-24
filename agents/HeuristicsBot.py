from ClueBasics.Player import Player
import random
import math
from fractions import Fraction
import logging

class HeuristicsBot(Player):

    def __init__(self, name, game, type):
        super().__init__(name, game, type)
        self.lastProcessedTurn = 0

    #------------------------------------------------------
    # Helpers

    def entropy(self, card):
        probs = (self.getProbability(owner, card) for owner in self.owners)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def highestSolutionChance(self, category):
        max_prob = max(self.getProbability("Solution", card) for card in category)
        tied_cards = [card for card in category if self.getProbability("Solution", card) == max_prob]
        max_entropy = max(self.entropy(card) for card in tied_cards)
        tied_entropy = [card for card in tied_cards if self.entropy(card) == max_entropy]
        return random.choice(tied_entropy)

    #------------------------------------------------------
    # Solution card tracking (ported from BayesianLearner)

    def removeCardFromPossibleCategories(self, card):
        """Remove a card from possible categories when we know a player has it (not in solution)"""
        if card.getType() == 'Suspect':
            if card in self.possibleSuspects:
                self.possibleSuspects.remove(card)
                logging.info(f"Removed {card} from possible suspects - a player has it")
        elif card.getType() == 'Weapon':
            if card in self.possibleWeapons:
                self.possibleWeapons.remove(card)
                logging.info(f"Removed {card} from possible weapons - a player has it")
        else:
            if card in self.possibleRooms:
                self.possibleRooms.remove(card)
                logging.info(f"Removed {card} from possible rooms - a player has it")

    def removeOtherCardsFromPossibleCategories(self, card):
        """Remove other cards from possible categories when we know one card is in the solution"""
        if card.getType() == 'Suspect':
            cards_to_remove = [otherCard for otherCard in self.possibleSuspects if otherCard != card]
            for otherCard in cards_to_remove:
                self.possibleSuspects.remove(otherCard)
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible suspects - {card} is the solution suspect")
        elif card.getType() == 'Weapon':
            cards_to_remove = [otherCard for otherCard in self.possibleWeapons if otherCard != card]
            for otherCard in cards_to_remove:
                self.possibleWeapons.remove(otherCard)
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible weapons - {card} is the solution weapon")
        else:
            cards_to_remove = [otherCard for otherCard in self.possibleRooms if otherCard != card]
            for otherCard in cards_to_remove:
                self.possibleRooms.remove(otherCard)
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible rooms - {card} is the solution room")

    def checkForSolutionCards(self):
        """Check for cards that must be in the solution or ruled out from solution"""
        for card in self.game.cards:
            solution_prob = self.getProbability("Solution", card)
            # Case 1: Solution probability is 0 - a player has it, remove from possible categories
            if solution_prob == 0:
                self.removeCardFromPossibleCategories(card)
            # Case 2: Solution probability is 1 - confirmed solution, remove other cards from category
            elif solution_prob >= 1.0:
                logging.info(f"Confirmed {card} is in the solution")
                self.removeOtherCardsFromPossibleCategories(card)
            # Case 3: No player can have the card - must be in the solution
            else:
                non_solution_total = sum(self.getProbability(owner, card)
                                        for owner in self.owners if owner != "Solution")
                if non_solution_total == 0 and solution_prob < 1.0:
                    logging.info(f"No player can have {card} - must be in solution")
                    for owner in self.owners:
                        if owner == "Solution":
                            self.setProbability(owner, card, 1.0)
                        else:
                            self.setProbability(owner, card, 0)
                    self.removeOtherCardsFromPossibleCategories(card)

    #------------------------------------------------------
    # Setup

    def initialCrossOff(self):
        self.createBeliefMatrix()
        # FIX: cross off own cards at startup so they're never considered as solution candidates
        for card in self.cards:
            self.crossOff(self, card)
        self.checkForSolutionCards()

    #------------------------------------------------------
    # Core belief updates

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

        for op in self.owners:
            self.setProbability(op, card, 0)

        self.setProbability(owner, card, 1)

        for other_card in self.ownersAndCards[owner]:
            if other_card == card:
                continue
            prob = self.ownersAndCards[owner][other_card]
            if prob > 0.04:
                self.setProbability(owner, other_card, prob - 0.04)
            else:
                self.setProbability(owner, other_card, prob / 2)

        # FIX: trigger cascading deductions after every crossOff
        self.checkForSolutionCards()

    def chooseSuggestion(self):
        suspect = self.highestSolutionChance(self.possibleSuspects)
        weapon = self.highestSolutionChance(self.possibleWeapons)
        room = self.highestSolutionChance(self.possibleRooms)
        return suspect, weapon, room

    def normalizeCardAcrossPlayers(self, card, players):
        fractions = []
        for player in players:
            prob = self.ownersAndCards[player][card]
            fractions.append((player, prob))

        total = sum(frac for _, frac in fractions)
        if total == 0:
            return

        for player, frac in fractions:
            new_frac = frac / total
            self.ownersAndCards[player][card] = new_frac

        # FIX: check for solution cards after every normalization
        self.checkForSolutionCards()

    #------------------------------------------------------
    # Suggestion processing

    def processNewSuggestions(self):
        public_log = self.game.getPublicSuggestionLog(self)
        for rec in public_log:
            if rec["turn"] <= self.lastProcessedTurn:
                continue

            suggester = rec["suggester"]
            responder = rec["responder"]
            suggestion_cards = rec["suggestion"]
            card_shown = rec["card_shown"]
            skipped_players = rec.get("skipped_players", [])

            if responder:
                if card_shown:
                    self.crossOff(responder, card_shown)
                    # Players who skipped can't have any of the suggested cards
                    for skipped_player in skipped_players:
                        for card in suggestion_cards:
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0)
                    # Increase chances for non-responder players on other cards
                    for owner in self.owners:
                        if owner != responder:
                            for card in self.game.cards:
                                if (card != card_shown and
                                        self.getProbability(owner, card) != 1 and
                                        self.getProbability(owner, card) != 0):
                                    prob = self.ownersAndCards[owner][card]
                                    if prob < 0.97:
                                        self.setProbability(owner, card, prob + 0.03)
                                    else:
                                        self.setProbability(owner, card, prob + (1 - prob) / 2)
                    self.normalizeCardAcrossPlayers(card_shown, self.owners)
                else:
                    # Players who skipped can't have any of the suggested cards
                    for skipped_player in skipped_players:
                        for card in suggestion_cards:
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0)
                        for card in self.game.cards:
                            if (self.getProbability(skipped_player, card) != 0 and
                                    self.getProbability(skipped_player, card) != 1):
                                prob = self.ownersAndCards[skipped_player][card]
                                if prob < 0.95:
                                    self.setProbability(skipped_player, card, prob + 0.05)
                                else:
                                    self.setProbability(skipped_player, card, prob + (1 - prob) / 2)
                        # FIX: normalize each suggested card excluding the skipped player
                        for card in suggestion_cards:
                            remaining_players = [p for p in self.owners if p != skipped_player]
                            self.normalizeCardAcrossPlayers(card, remaining_players)

                    self.checkForSolutionCards()

            else:
                # No one responded - cards must be with suggester or in solution
                for card in suggestion_cards:
                    for player in self.owners:
                        if player not in [suggester, 'Solution']:
                            self.setProbability(player, card, 0)
                    self.normalizeCardAcrossPlayers(card, [suggester, 'Solution'])

                self.checkForSolutionCards()

            self.privateSuggestionLog.append(rec)
            self.runBackwardInference()
            self.lastProcessedTurn = rec['turn']

    def runBackwardInference(self):
        for rec in self.privateSuggestionLog:
            responder = rec["responder"]
            if not responder or rec["card_shown"]:
                continue

            suggestion_cards = rec["suggestion"]
            turn = rec["turn"]

            possible_cards = []
            for card in suggestion_cards:
                prob = self.ownersAndCards[responder][card]
                if prob > 0:
                    possible_cards.append(card)

            if len(possible_cards) == 1:
                shown_card = possible_cards[0]
                rec["card_shown"] = shown_card
                if self.game.verbose:
                    print(f"Turn {int(turn / len(self.players))}: Deduced {responder} showed {shown_card}")
                logging.info(f"Turn {int(turn / len(self.players))}: Deduced {responder} showed {shown_card}")
                self.crossOff(responder, shown_card)

                for owner in self.owners:
                    if owner != responder:
                        for card in self.game.cards:
                            if (card != shown_card and
                                    self.getProbability(owner, card) != 1 and
                                    self.getProbability(owner, card) != 0):
                                prob = self.ownersAndCards[owner][card]
                                if prob < 0.97:
                                    self.setProbability(owner, card, prob + 0.03)
                                else:
                                    self.setProbability(owner, card, prob + (1 - prob) / 2)

                self.normalizeCardAcrossPlayers(shown_card, self.owners)

    #------------------------------------------------------
    # Turn

    def playTurn(self):
        if not self.inGame:
            return

        self.processNewSuggestions()
        perp, weapon, room = self.chooseSuggestion()
        responder, shown_card = self.game.makeSuggestion(self, perp, weapon, room)

        if responder is not None:
            if self.game.verbose:
                print(f"{responder.name} showed a card - {shown_card.name}.")
            self.crossOff(responder, shown_card)
            # FIX: use shown_card (not card) to avoid variable shadowing bug
            for owner in self.owners:
                if owner != responder:
                    for card in self.game.cards:
                        if (card != shown_card and
                                self.getProbability(owner, card) != 1 and
                                self.getProbability(owner, card) != 0):
                            prob = self.ownersAndCards[owner][card]
                            if prob < 0.97:
                                self.setProbability(owner, card, prob + 0.03)
                            else:
                                self.setProbability(owner, card, prob + (1 - prob) / 2)
            self.normalizeCardAcrossPlayers(shown_card, self.owners)
            self.processNewSuggestions()

        if responder is None:
            if perp not in self.cards and weapon not in self.cards and room not in self.cards:
                if self.makeAccusation(perp, weapon, room):
                    return self.name

        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                return self.name
