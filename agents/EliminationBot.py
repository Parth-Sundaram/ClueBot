from ClueBasics.Player import Player
import random

class EliminationBot(Player):

    def chooseSuggestion(self):
        suspect = random.choice(self.possibleSuspects)
        weapon = random.choice(self.possibleWeapons)
        room = random.choice(self.possibleRooms)
        return suspect, weapon, room

    def playTurn(self):
        if not self.inGame:
            return

        perp, weapon, room = self.chooseSuggestion()
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        #Eliminates the new card
        if not owner is None:
            if self.game.verbose:
                print(f"{owner.name} showed a card.")
            self.crossOff(owner, card)

        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                if self.game.verbose:
                    print(f"{self.name} WINS! The solution was correct.")
                return self.name
            else:
                if self.game.verbose:
                    print(f"{self.name} made a wrong accusation and is out.")
