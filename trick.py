from deck import Deck

class Trick:
    def __init__(self, dealer, players):
        self.deck = Deck()
        self.players = players
        self.numplayers = 4
        
        #Deal out all the cards + the kitty
        self.players[0].hand = self.deck.deal(5)
        self.players[1].hand = self.deck.deal(5)
        self.players[2].hand = self.deck.deal(5)
        self.players[3].hand = self.deck.deal(5)
        
        self.players[dealer].isdealer = True
        self.kitty = self.deck.deal(1)
        print("Your hand: " )
        print(self.players[0].hand)
        print("The kitty: ")
        print(self.kitty)

    def trumpSet (self, dealer):
        x = dealer + 1
        while True:
            if (x > 3):
                x = 0
            z = self.players[x].getChoice()
            if z:
                if self.players[x].isdealer:
                    print ("Player " + str(x+1) + " picks up the card.")
                else:
                    print("Player " + str(x+1) + " orders the dealer up.")
                    if self.players[x].team == self.players[dealer].team:
                        self.players[dealer].playing = False
                        print("Player " + str(x + 1) + " is going alone.")
                        self.numplayers = 3
                self.players[dealer].swapCard(self.kitty[0])
                self.deck.setTrump(self.kitty)
                break
            else:
                print("Player " + str(x+1) + " passes.")
            if x == dealer:
                break
            x += 1
        if self.deck.trumpset == False:
            #If nobody ordered it up, go around to see if someone wants to pick trump
            x = dealer + 1
            while True:
                if (x > 3):
                    x = 0
                z = self.players[x].callTrump()
                if z:
                    self.deck.setTrump(z)
                    a = self.players[x].goAlone()
                    if a:
                        #Calculate the teammate here. If it's player 0, m will be 2. If player 1, m will be 3
                        #If player 2, m will be 4, so we set it to 0, the true partner. If player 3, m will be 5.
                        m = x + 2
                        if m == 4:
                            m = 0
                        elif m == 5:
                            m = 1
                        self.players[m].playing = False
                        self.numplayers = 3
                    break
                else:
                    print("Player " + str(x+1) + " passes.")
                if x == dealer:
                    break
                x += 1
        if self.deck.trumpset == False:
            print ("Stalled trick. Time to re-shuffle and start over!")
            return 0
        else:
            return True

    def playTrick (self, dealer):
        #Begin main play
        lead = dealer + 1
        for x in range(5):
            w = self.playRound(lead)
            lead = w
            self.players[w].rounds += 1
            print("Player " + str(w + 1) + " wins!")
        t1 = self.players[0].rounds + self.players[2].rounds
        t2 = self.players[1].rounds + self.players[3].rounds

        #This section scores the trick - 4 points if alone + 5 rounds won
        #2 points if 5 rounds won not alone
        #1 point otherwise
        if t1 == 5:
            if (self.players[0].playing == False) or (self.players[2].playing == False):
                o = [4, 0]
            else:
                o = [2, 0]
        elif t2 == 5:
            if (self.players[1].playing == False) or (self.players[3].playing == False):
                o = [0, 4]
            else:
                o = [0, 2]
        elif t1 > t2:
            o = [1, 0]
        elif t2 > t1:
            o = [0, 1]

        return o
    
    def playRound(self, l):
        x = l
        pile = []
        win_pts = 0
        win_plyr = 0
        while True:
            if len(pile) >= self.numplayers:
                break
            if x == 4:
                x = 0        
            if self.players[x].playing:
                if len(pile) > 0:
                    move = self.players[x].getMove(pile, self.deck.trump, self.deck)
                else:
                    move = self.players[x].getMove(0, self.deck.trump, self.deck)

                pile.append(move)
                if (move[-1] in pile[0]) or (move[-1] in self.deck.trump) or (self.deck.lbower in move):
                    if self.deck.cardrank.get(move) > win_pts:
                        win_pts = self.deck.cardrank.get(move)
                        win_plyr = x
                print("Player " + str(x+1) + ": " + move + "| Hand: " + format(self.players[x].hand))
            else:
                print("Player " + str(x+1) + " is not playing.")
            x += 1
        return win_plyr