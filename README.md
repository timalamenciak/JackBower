# JackBower
This is an implementation of Canadian-rules Euchre with a Deep Q Network AI using PyTorch. Code was inspired by Jacob Miske's [Michigander]([https://github.com/JacobMiske/michigander]) Euchre engine.

## OOEuchre_core
The core Euchre engine allows for human and computer players. The computer players operate on one of two strategies: random cards or highest card. This is the basis for the Deep Q engine. The core code contains print statements that show the pile and hand of Player 1. 

## OOEuchre_AI
This script implements a DeepQ network in the game of Euchre. The code trains the network on each player's moves. The state-action pair is the pile and trump paired with the card played. Teams are ignored and the network's goal is to maximize the amount of rounds won. There are 5 rounds in a trick.

I have found that the AI needs about 30,000 games before it starts to win against a random strategy. This may be because of the sizeable number of possible states. A pre-trained model has also been included.
