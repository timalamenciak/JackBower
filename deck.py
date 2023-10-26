import random

class Deck:    
    def __init__(self):
      # creates a Euchre deck. Cards are in order - shuffling happens from deal.
        self.suits = ['H', 'S', 'C', 'D']
        values = ['9', '10', 'J', 'Q', 'K', 'A']
    
        A = [[[x,y] for x in values] for y in self.suits]
        self.deck = [item for subl in A for item in subl]
        self.deck = [''.join(i) for i in self.deck]
        self.isdealer = False
        self.trump = ""
        self.lbower = ""
        self.trumpset = False
        
        #This sets the starting rank for each card. There is almost certainly a more elegant way to do this.
        #This ranking gets updated when trump is set.
        self.cardrank = {'AC': 6, 'AD': 6, 'AS': 6, 'AH': 6, '9C': 1, '9D': 1, '9S': 1, '9H': 1,
            '10C': 2, '10D': 2, '10S': 2, '10H': 2, 'JC': 3, 'JD': 3, 'JS': 3, 'JH': 3, 'QC': 4, 'QD': 4, 'QS': 4, 'QH': 4,
            'KC': 5, 'KD': 5, 'KS': 5, 'KH': 5}
    
    def deal(self, numcards):
        # deals a specified number of cards randomly from deck.
        hand = []
        while len(hand) < numcards:
            A = self.deck[random.randint(0, len(self.deck)-1)]
            hand.append(A)
            self.deck.remove(A)
        return hand
    
    def setTrump(self, card):
        # sets the trump according to a card string (e.g. 10S)
        # this updates the rankings, trumps get +10
        self.trump = card[-1]
        for x in self.cardrank:
            if x[-1] in self.trump[-1]:
                v = self.cardrank.get(x) + 10
                self.cardrank.update({x : v})
        # Have to set the bowers here (Jack of suit is 18, jack of other colour is 17)
        if "C" in self.trump:
            self.cardrank.update({"JC" : 18})
            self.cardrank.update({"JS" : 17})
            self.lbower = "JS"
        if "S" in self.trump: 
            self.cardrank.update({"JS" : 18})
            self.cardrank.update({"JC" : 17})
            self.lbower = "JC"
        if "D" in self.trump:
            self.cardrank.update({"JD" : 18})
            self.cardrank.update({"JH" : 17})
            self.lbower = "JH"
        if "H" in self.trump: 
            self.cardrank.update({"JH" : 18})
            self.cardrank.update({"JD" : 17})
            self.lbower = "JD"
        print("Trump is " + self.trump[-1])
        self.trumpset = True    
