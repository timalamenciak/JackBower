import random

class Player:
    def __init__(self, type, team):
        self.type = type
        self.hand = []
        self.isdealer = False
        self.team = team
        self.rounds = 0
        self.playing = True
    
    def getChoice(self):
        if (self.type == "C") or (self.type == "CH"):
            #Computer choice goes in here
            #For now, simple AI is coin flip -- 50/50 chance:
            r = random.randint(0,1)
            if r == 0:
                return False
            else:
                return True
        if self.type == "H":
            #Human choice
            if self.isdealer:
                choice = input("Do you want to pick it up? (Y/N)")
                if choice == "Y":
                    return True
                if choice == "N":
                    return False
            choice = input("Do you want to order the dealer up?")
            if choice == "Y":
                return True
            if choice == "N":
                return False

    def callTrump (self):
        if (self.type == "C") or (self.type == "CH"):
            #Computer choice goes in here
            #For now, simple AI is coin flip -- 50/50 chance:
            if random.randint(0,1) == 1:
                c = ["H", "D", "S", "C"]
                return c[random.randint(0,3)]
            else:
                return 0
        if self.type == "H":
            #Human choice
            while True:
                choice = input("Do you want to call trump? (C/S/D/H or P for pass)")
                if choice == "P":
                    return 0
                elif (choice == "H" or choice == "D" or choice == "S" or choice == "C"):
                    return choice
                else: 
                    print ("Invalid input")

    def goAlone (self):
        if self.type == "C":
            return random.randint(0,1)
        elif self.type == "H":
            a = input("Do you want to go alone? (Y/N)")
            if a == "Y":
                return True
            else:
                return False

    def getMove (self, pile, trump, deck):
        if pile == 0:
            if self.type == "C":
                #True random agent
                c = str(self.hand[random.randint(0,len(self.hand)-1)])
                self.hand.remove(c)
                return c
            if self.type == "CH":
                #High card Harry plays the highest value card
                pts2 = 0 
                for x in self.hand:
                    pts1 = deck.cardrank.get(x)
                    if pts1 > pts2:
                        pts2 = pts1
                        c = x
                self.hand.remove(c)
                return c
            if self.type == "H":
                while True:
                    c = input("Which card do you want to lead with? (Hand: " + str(self.hand) + ")")
                    if c in self.hand:
                        self.hand.remove(c)
                        return c
                    else:
                        print ("That card isn't in your hand")
        if self.type == "CH":
            c = self.getValidMove(pile[0], trump)
            pts2 = 0
            for x in c:
                pts1 = deck.cardrank.get(x)
                if pts1 > pts2:
                    pts2 = pts1
                    card = x
            self.hand.remove(card)
            return card
        if self.type == "C":
            c = self.getValidMove(pile[0], trump)
            card = c[random.randint(0,len(c)-1)]
            self.hand.remove(card)
            return card
        if self.type == "H":
            while True:
                    c = input("Which card do you want to play? Valid moves: {}" . format(self.getValidMove(pile[0], trump)))
                    if c in self.hand:
                        self.hand.remove(c)
                        return c
                    else:
                        print ("That card isn't in your hand")
        
    def getValidMove(self, lead, trump):
        #Valid moves are the suit that's led or trump if no suit that is led. 
        #If none of above is true, any card is valid but will not count for points.
        moves = []

        # The left bower gets funky because it is the trump suit for the purposes of the round
        lbower = ""
        if trump[-1] == "C":
            lbower = "JS"
        elif trump[-1] == "S":
            lbower = "JC"
        elif trump[-1] == "D":
            lbower = "JH"
        elif trump[-1] == "H":
            lbower = "JD"

        for x in self.hand:
            if lead[-1] in x:
                #Check if it's a jack and the suit is the bower - if so, we skip.
                if (x[0] == "J") and (lbower in lead[-1]):
                    pass
                else:
                    moves.append(x)
            #Check if the lead suit is the trump suit, and if the x happens to be the lbower
            if (lead[-1] in trump) and (x == lbower):
                moves.append(x)

        if len(moves) == 0:
            moves = self.hand
        
        return moves
        
    def swapCard (self, card):
        if self.type == "C":
            self.hand.remove(self.hand[random.randint(0,len(self.hand)-1)])
            self.hand.append(card)
        if self.type == "H":
            while True:
                swchoice = input("Enter the card from your hand to discard: ")
                if swchoice in self.hand:
                    break
                print("That card is not in your hand. Try again.")
            self.hand.remove(swchoice)
            self.hand.append(card)
        return True
