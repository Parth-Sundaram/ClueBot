import time
import random
from .Card import Card

class GameRules:
    totalCards = 21
    ROOMS = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    SUSPECTS = ["Colonel Mustard", "Reverend Green", "Miss Scarlet", "Mrs. Peacock", "Mrs White", "Professor Plum"]
    WEAPONS = ["Knife", "Candlestick", "Rope", "Revolver", "Wrench", "Lead Pipe"]

    def __init__(self, players, verbose=False):
        self.players = players
        self.verbose = verbose
        self.suspectCards = {}
        self.weaponCards = {}
        self.roomCards = {}
        self.deck = {}
        self.cards = []
        self.suggestionLog = []
        self.turn = 0
        self.gameTurn = 0
        self.hasHuman = False

        for card in self.SUSPECTS:
            suspect = Card("Suspect", card)
            self.suspectCards[card] = suspect
            self.deck[card] = suspect
            self.cards.append(suspect)

        for card in self.WEAPONS:
            weapon = Card("Weapon", card)
            self.weaponCards[card] = weapon
            self.deck[card] = weapon
            self.cards.append(weapon)

        for card in self.ROOMS:
            room = Card("Room", card)
            self.roomCards[card] = room
            self.deck[card] = room
            self.cards.append(room)

        self.solution = {
            "Suspect": random.choice(list(self.suspectCards.values())),
            "Weapon":  random.choice(list(self.weaponCards.values())),
            "Room":    random.choice(list(self.roomCards.values()))
        }

        self.deck.pop(self.solution["Suspect"].getName())
        self.deck.pop(self.solution["Weapon"].getName())
        self.deck.pop(self.solution["Room"].getName())

    def makeAccusation(self, player, perp, weapon, room):
        if self.verbose:
            print(f"{player.name} accuses {perp.name} with a {weapon.name} in the {room.name}")
        if (self.solution["Suspect"] == perp and
                self.solution["Weapon"] == weapon and
                self.solution["Room"] == room):
            if self.verbose:
                print(f"{player.name} has won!")
            return True
        if self.verbose:
            print(f"{player.name} has accused wrong and is out")
        player.inGame = False
        return False

    def makeSuggestion(self, player, perp, weapon, room, reveal_chooser=None):
        """
        reveal_chooser: optional callable(matching_cards) -> card
            Injected by the RL environment so the DQN reveal head can choose
            which card the RL agent shows when it is the responder.
            All other players always use their own showCard() logic.
        """
        if self.verbose:
            print(f"{player.name} suggests: {perp} with {weapon} in {room}")
        self.turn += 1
        suggestionCards = [perp, weapon, room]
        playerPos = self.players.index(player)
        i = (playerPos + 1) % len(self.players)

        suggestion_record = {
            "turn": self.turn,
            "suggester": player,
            "suggestion": suggestionCards,
            "ambiguous": True,
            "responder": None,
            "card_shown": None,
            "possible_shown_cards": set(suggestionCards),
            "skipped_players": []
        }

        while i != playerPos:
            responding_player = self.players[i]
            # Only inject the reveal_chooser for the RL agent itself
            chooser = reveal_chooser
            cardShown = responding_player.refuteSuggestion(suggestionCards, reveal_chooser=chooser)

            if cardShown is not None:
                suggestion_record["responder"] = responding_player
                suggestion_record["card_shown"] = cardShown
                self.suggestionLog.append(suggestion_record)
                return responding_player, cardShown

            suggestion_record["skipped_players"].append(responding_player)
            i = (i + 1) % len(self.players)

        self.suggestionLog.append(suggestion_record)
        return None, None

    def getPublicSuggestionLog(self, player):
        public_log = []
        for rec in self.suggestionLog:
            entry = {
                "turn": rec["turn"],
                "suggester": rec["suggester"],
                "suggestion": list(rec["suggestion"]),
                "responder": rec["responder"] if rec["responder"] else None,
                "card_shown": None,
                "ambiguous": rec["ambiguous"],
                "skipped_players": rec.get("skipped_players", [])
            }
            if player == rec["suggester"] or player == rec["responder"]:
                entry["card_shown"] = rec["card_shown"]
                entry["ambiguous"] = False
            public_log.append(entry)
        return public_log

    def dealCards(self):
        deckCards = list(self.deck.keys())
        random.shuffle(deckCards)
        playerIter = 0
        for card in deckCards:
            self.players[playerIter].isDealt(self.deck[card])
            playerIter += 1
            if playerIter == len(self.players):
                playerIter = 0

        for player in self.players:
            if self.verbose:
                print(f"{player.name} has {player.numCards} cards")
            player.initialCrossOff()
            if player.type == "Human":
                player.revealCards()

    def checkAllPlayers(self):
        return sum(1 for p in self.players if p.inGame) > 1

    def findWinner(self):
        for player in self.players:
            if player.inGame:
                return player

    def gameLoop(self):
        self.dealCards()
        while True:
            self.gameTurn += 1
            if self.verbose:
                print(f"Turn {self.gameTurn}")
            if not self.checkAllPlayers():
                break
            for player in self.players:
                winner = player.playTurn()
                if winner:
                    return
            time.sleep(2)
