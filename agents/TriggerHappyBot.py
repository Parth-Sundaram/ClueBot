from ClueBasics.Player import Player
import random

class TriggerHappyBot(Player):

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

        if owner is None:
            if self.game.verbose:
                print(f"No one disproved. {self.name} might try an accusation!")
            if self.makeAccusation(perp, weapon, room):
                if self.game.verbose:
                    print(f"{self.name} WINS! The solution was correct. ")
                return self.name
            else:
                if self.game.verbose:
                    print(f"{self.name} made a wrong accusation and is out.")
                self.inGame = False
        else:
            if self.game.verbose:
                print(f"{owner.name} showed a card.")
