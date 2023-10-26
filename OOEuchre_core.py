# This is a barebones Euchre engine for Canadian Euchre, based on Jacob Miske's Python Euchre game.
# Tim Alamenciak - MIT License
# For use with Deep Q Learning
# Version 0.0.1

from player import Player
from trick import Trick

#Code for the actual game engine

#Initialize the lists of scores
t1scores = []
t2scores = []

for x in range(1):
    players = []
    players.append(Player("C",1))
    players.append(Player("C",2))
    players.append(Player("C",1))
    players.append(Player("C",2))

    team1score = 0
    team2score = 0  

    q = 0
    dealer = 0

    while True:
        print ("TRICK " + str(q))
        print ("DEALER: Player " + str(dealer+1))
        for x in range(4):
            players[x].playing = True
            players[x].isdealer = False
            players[x].rounds = 0
        players[dealer].isdealer = True
        trick = Trick(dealer, players)
        
        if trick.trumpSet(dealer):
            points = trick.playTrick(dealer)
            team1score += points[0]
            team2score += points[1]

        if team1score >= 10:
            print("T1: " + str(team1score))
            print("T2: " + str(team2score))
            t1scores.append(team1score)
            t2scores.append(team2score)
            print("Team 1 wins!")
            break
        elif team2score >= 10:
            print("T1: " + str(team1score))
            print("T2: " + str(team2score))
            t1scores.append(team1score)
            t2scores.append(team2score)        
            print("Team 2 wins!")
            break
    dealer += 1
    if dealer == 4:
        dealer = 0
    q += 1