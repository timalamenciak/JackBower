# This is a Euchre engine for Canadian Euchre, based on Jacob Miske's python Euchre game.
# Tim Alamenciak - MIT License
# For use with Deep Q Network learning 

import random
random.seed(10)
import torch
import numpy as np
import pandas as pd
from collections import deque
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Deck:    
    def __init__(self):
      # creates a Euchre deck. Cards are in order - shuffling happens from deal.
        suits = ['H', 'S', 'C', 'D']
        values = ['9', '10', 'J', 'Q', 'K', 'A']
    
        A = [[[x,y] for x in values] for y in suits]
        self.deck = [item for subl in A for item in subl]
        self.deck = [''.join(i) for i in self.deck]
        self.deckindex = {'9H':1, '10H':2, 'JH':3, 'QH':4, 'KH':5, 'AH':6, '9S':7, '10S':8, 'JS':9, 'QS':10, 'KS':11, 'AS':12, '9C':13, '10C':14, 'JC':15, 'QC':16, 'KC':17, 'AC':18, '9D':19, '10D':20, 'JD':21, 'QD':22, 'KD':23, 'AD':24}    #We need this index because we take cards out of deck
        self.isdealer = False
        self.trump = ""
        self.lbower = ""
        self.trumpset = False
        
        #This sets the starting rank for each card. There is almost certainly a more elegant way to do this.
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
        #print("Trump is " + self.trump[-1])
        self.trumpset = True    

class Player:
    def __init__(self, type, team, model):
        self.type = type
        self.hand = []
        self.isdealer = False
        self.team = team
        self.rounds = 0
        self.totalrounds = 0
        self.playing = True
        self.deckindex = {'9H':1, '10H':2, 'JH':3, 'QH':4, 'KH':5, 'AH':6, '9S':7, '10S':8, 'JS':9, 'QS':10, 'KS':11, 'AS':12, '9C':13, '10C':14, 'JC':15, 'QC':16, 'KC':17, 'AC':18, '9D':19, '10D':20, 'JD':21, 'QD':22, 'KD':23, 'AD':24}    #We need this index because we take cards out of deck

        self.state = [] #This will store the state at the time a player makes a move
        self.action = 0 #This will store the action the player takes
        self.reward = 0 #This will store the reward - for now, 0.
        if model != 0:
            self.model = model
        if type == "AI":
            self.n_games = 0
            self.epsilon = 0 #Randomness
            self.gamma = 0.9 #Discount rate
            self.memory = deque(maxlen=MAX_MEMORY) 


    
    def getChoice(self):
        if (self.type == "C") or (self.type == "CH") or (self.type == "AI"):
            #Computer choice goes in here
            #For now, simple AI is coin lord -- 50/50 chance:
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
        if (self.type == "C") or (self.type == "CH") or self.type == "AI":
            #Computer choice goes in here
            #For now, simple AI is coin lord -- 50/50 chance:
            if random.randint(0,1) == 1:
                c = ["H", "D", "S", "C"]
                return c[random.randint(0,3)]
            else:
                return 0
            #return random.randint(0,1)
            #For testing - pass:
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
        #Store the state, which will be used to train the Q engine
        self.state = self.ai_get_state(trump, pile)
        if pile == 0:
            #TODO: Rewrite this function to be more efficient by just setting moves = self.hand if pile is 0
            if self.type == "C":
                #True random agent
                c = str(self.hand[random.randint(0,len(self.hand)-1)])
                self.hand.remove(c)
                return c
                #This is the first place the Q network will hook in
                #State: Which card(s) have been played? What is trump? What are choices?
                #We need to pass it pile, trump and self.player.hand
                #Then we need to reinforce it with whether it won the round or not
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
            if self.type == "AI":
                #Code to pass the state to AI
                #Created a fixed deck - self.deck.deckindex
                
                #For testing, true random:
                #Here we're going to have to pull the Q scores, play the card with highest
                #We're going to have to get the valid moves and pick the valid move with highest Q
                actions = self.ai_get_actions(self.state)
                actions = actions.detach().numpy()
                # Now we need to subset actions to include the cards that are a valid move, and check which
                # is the biggest!
                #TODO
                
                hs = None
                for x in self.hand:
                    n = self.deckindex.get(x)
                    if hs == None:
                        hs = actions[n-1]
                        c = str(x)
                    elif actions[n-1] > hs:
                        hs = actions[n-1]
                        c = str(x)
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
        if self.type == "AI":
            actions = self.ai_get_actions(self.state)
            actions = actions.detach().numpy()
            mvs = self.getValidMove(pile[0], trump)

            hs = None
            for x in mvs:
                n = self.deckindex.get(x)
                if hs == None:
                    hs = actions[n-1]
                    c = str(x)
                elif actions[n-1] > hs:
                    hs = actions[n-1]
                    c = str(x)
            self.hand.remove(c)
            return c

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
    
    def ai_get_state(self, trump, pile):
        s = [0, 0, 0, 0, 0]
        card_dict_flip = {'9H':1, '10H':2, 'JH':3, 'QH':4, 'KH':5, 'AH':6, '9S':7, '10S':8, 'JS':9, 'QS':10, 'KS':11, 'AS':12, '9C':13, '10C':14, 'JC':15, 'QC':16, 'KC':17, 'AC':18, '9D':19, '10D':20, 'JD':21, 'QD':22, 'KD':23, 'AD':24}
        #TODO: Fix this section to handle piles of different lengths
        #match up each card with its dict value
        n = 0
        if (pile != 0):
            for x in pile:
                s[n] = card_dict_flip[x]
                n += 1
            while n < 4:
                s[n] = 0
                n += 1

        if "H" in trump:
            s[4] = 1
        elif "C" in trump:
            s[4] = 2
        elif "D" in trump:
            s[4] = 3
        elif "S" in trump:
            s[4] = 4
        
        #This should maybe return a numpy array rather than a pandas dataframe.

        return(s)

    def ai_remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def ai_train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #List of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def ai_train_short_memory(self, state, action, reward):
        self.trainer.train_step(state, action, reward)

    def ai_get_actions(self, state):
        # Tradeoff between exploration and exploitation
        # I reason that the random agents (p2,3,4) are doing enough exploration
        # so this function just returns the Q values.
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        return prediction
    
class Trick:
    def __init__(self, dealer, players):
        self.deck = Deck()
        self.players = players
        self.numplayers = 4
        
        #Initialize the hand lists so we can get training data
        #each hand is a list of lists, the sublist is: pile 1, pile 2, pile 3, pile 4, trump, action, reward
        #TODO: There's going to be some weirdness here if a player isn't playing.
        self.hand1 = []
        self.hand2 = []
        self.hand3 = []
        self.hand4 = []


        #Deal out all the cards + the kitty
        self.players[0].hand = self.deck.deal(5)
        self.players[1].hand = self.deck.deal(5)
        self.players[2].hand = self.deck.deal(5)
        self.players[3].hand = self.deck.deal(5)
        self.players[dealer].isdealer = True
        self.kitty = self.deck.deal(1)

        #Choice round - each player can order the dealer up, and last the dealer can choose to pick up
        #IRL it starts with the player to the right of the dealer.

    def trumpSet (self, dealer):
        x = dealer + 1
        while True:
            if (x > 3):
                x = 0
            z = self.players[x].getChoice()
            if z:
                if self.players[x].isdealer:
                    #print ("Player " + str(x+1) + " picks up the card.")
                    pass
                else:
                    #print("Player " + str(x+1) + " orders the dealer up.")
                    if self.players[x].team == self.players[dealer].team:
                        self.players[dealer].playing = False
                        #print("Player " + str(x + 1) + " is going alone.")
                        self.numplayers = 3
                self.players[dealer].swapCard(self.kitty[0])
                self.deck.setTrump(self.kitty)
                #Code here to make decision as to whether going alone or normal
                #e.g. check if dealer is on team, or if person is dealer = go alone
                #otherwise, normal play
                break
            else:
                #print("Player " + str(x+1) + " passes.")
                pass
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
                    #print("Player " + str(x+1) + " passes.")
                    pass
                if x == dealer:
                    break
                x += 1
        if self.deck.trumpset == False:
            #print ("Stalled trick. Time to re-shuffle and start over!")
            return 0
        else:
            return True

    def playTrick (self, dealer):
        #Now the kitty has been dealt with, trump has been called
        #Begin main play
        lead = dealer + 1


        for x in range(5):
            w = self.playRound(lead)
            #TODO: Here is where we need to train the AI because we know who won the round. Player
            #class will know what the state was at the time of the play, the move.
            lead = w
            self.players[w].rounds += 1
            self.players[w].reward = 1
            #print("Player " + str(w + 1) + " wins!")
            
            #There is almost definitely a more efficient way to accomplish the below, but I can't think of it.
            #We need to build a list here out of the state, action and reward variables
            newhand = self.players[0].state
            newhand.append(self.players[0].action)
            newhand.append(self.players[0].reward)

            #Add that list to the hand list of lists
            self.hand1.append(newhand)
            #Repeat for all players
            newhand = self.players[1].state
            newhand.append(self.players[1].action)
            newhand.append(self.players[1].reward)

            #Add that list to the hand list of lists
            self.hand2.append(newhand)

            newhand = self.players[2].state
            newhand.append(self.players[2].action)
            newhand.append(self.players[2].reward)

            #Add that list to the hand list of lists
            self.hand3.append(newhand)

            newhand = self.players[3].state
            newhand.append(self.players[3].action)
            newhand.append(self.players[3].reward)

            #Add that list to the hand list of lists
            self.hand4.append(newhand)

            #Reset reward
            self.players[w].reward = 0
        
        #Here's where we train the AI
        # We are essentially ignoring the trick scoring and training it based on whether the action won
        # the given round. The Q score will factor in future round potentials. This may be a point we 
        # need to revise later.
        # We have four lists - hand1 through hand4. Each contains the pile at the time of the move, trump, 
        # the action taken and the reward (1 for win, 0 for loss)    

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

                #TODO: Change this to pull the card score and see if it is highest in player's hand
                #validmvs = self.players[x].getValidMove(pile[0], self.deck.trump)
                #for z in validmvs:
                #    if self.deck.cardrank.get(move) < self.deck.cardrank.get(z):
                #        self.players[x].action = [0,1]
                #    else:
                #        self.players[x].action = [1,0]
                
                
                self.players[x].action = self.deck.deckindex.get(move) #This converts the card to an int
                pile.append(move)
                if (move[-1] in pile[0]) or (move[-1] in self.deck.trump) or (self.deck.lbower in move):
                    if self.deck.cardrank.get(move) > win_pts:
                        win_pts = self.deck.cardrank.get(move)
                        win_plyr = x
                #print("Player " + str(x+1) + ": " + move + "| Hand: " + format(self.players[x].hand))
            else:
                #print("Player " + str(x+1) + " is not playing.")
                pass
            x += 1
        return win_plyr

#Code for the actual game engine
#Right now I need to hard code the number of games into the engine so that the Agent can 
#stay the same throughout and learn. It's not the cleanest, but it gets us there I think.
class Euchre:
    def __init__(self, trainer, model, train):
        self.model = model
        self.train = train
        self.reset()
        dealer = 0
        while True:
            #print ("DEALER: Player " + str(dealer+1))
            for x in range(4):
                self.players[x].playing = True
                self.players[x].isdealer = False
                self.players[x].totalrounds += self.players[x].rounds
                self.players[x].rounds = 0
            self.players[dealer].isdealer = True
            trick = Trick(dealer, self.players)

            if trick.trumpSet(dealer):
                points = trick.playTrick(dealer)
                #Train the AI off the result of the tricks
                #We have to check if each player is playing first, otherwise we'd be training on bum data
                if train == 1:
                    if self.players[0].playing:
                        trainer.train_step(trick.hand1)
                    if self.players[1].playing:
                        trainer.train_step(trick.hand2)
                    if self.players[2].playing:
                        trainer.train_step(trick.hand3)
                    if self.players[3].playing:
                        trainer.train_step(trick.hand4)
                
                self.team1score += points[0]
                self.team2score += points[1]
            if self.team1score >= 10:
                #print("T1: " + str(self.team1score))
                #print("T2: " + str(self.team2score))
                #print("Team 1 wins!")
                break
            elif self.team2score >= 10:
                #print("T1: " + str(self.team1score))
                #print("T2: " + str(self.team2score))
                #print("Team 2 wins!")
                break
            dealer += 1
            if dealer == 4:
                dealer = 0
        

    def reset(self):
        self.players = []
        self.players.append(Player("C",1,0))
        self.players.append(Player("AI",2,self.model))
        self.players.append(Player("C",1,0))
        self.players.append(Player("C",2,0))
        self.team1score = 0
        self.team2score = 0  



model = Linear_QNet(5, 10, 24) #NEED TO ADJUST SIZES FOR EUCHRE
trainer = QTrainer(model, lr=0.001, gamma=0.5) #TODO        

#Let's track the round wins
p1wins = []
p2wins = [] 
p3wins = []
p4wins = []
p1avg = []
p2avg = []
p3avg = []
p4avg = []

#Training on 20,000 games resulted in AI outperforming random across 10 games
#Load the trained model:
model.load_state_dict(torch.load('model/20Kgames_model.pth'))
model.eval()

for x in range(10000):
    print("Game: ", x)
    #Here you set how many games to train on vs. test
    if x < 9990:
        euchre = Euchre(trainer, model, train = 1)
    else:
        euchre = Euchre(trainer, model, train = 0)
    p1wins.append(euchre.players[0].totalrounds)
    p1avg.append(sum(p1wins)/len(p1wins))
    p2wins.append(euchre.players[1].totalrounds)
    p2avg.append(sum(p2wins)/len(p2wins))
    p3wins.append(euchre.players[2].totalrounds)
    p3avg.append(sum(p3wins)/len(p3wins))
    p4wins.append(euchre.players[3].totalrounds)
    p4avg.append(sum(p4wins)/len(p4wins))


model.save()
import matplotlib.pyplot as plt
plt.plot(p1wins, label="Player 1")
plt.plot(p2wins, label="AI")
plt.plot(p3wins, label="Player 3")
plt.plot(p4wins, label="Player 4")
plt.legend()
plt.ylabel('Rolling average rounds won')
plt.xlabel('games')
plt.title("Rounds won per game")
plt.show()

print("Average wins")
print(sum(p1wins)/len(p1wins))
print(sum(p2wins)/len(p2wins))
print(sum(p3wins)/len(p3wins))
print(sum(p4wins)/len(p4wins))
print("Last 10 games average")
print(sum(p1wins[len(p1wins)-10:len(p1wins)])/10)
print(sum(p2wins[len(p2wins)-10:len(p2wins)])/10)    
print(sum(p3wins[len(p3wins)-10:len(p3wins)])/10)
print(sum(p4wins[len(p4wins)-10:len(p4wins)])/10)


