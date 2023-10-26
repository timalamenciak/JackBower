import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__ (self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, hand):
        #Here we should just take one variable - hand, which is a list of lists.
        # Each sub-list in hand contains 7 ints: pile1, pile2, pile3, pile4, trump, action, reward
        # Our AI should take pile1, pile2, pile3, pile4, trump (states), and predict the Q score for actions
        # We will train based on the action-reward combo, and the state value of the next state (e.g. hand[n+1])
        h = np.array(hand)
        
        #Here we pull states, actions and rewards out. We only ever train on a full trick, which is 5 cards.
        states = torch.tensor(h[0:5,0:5], dtype=torch.float)
        actions = torch.tensor(h[0:5,5], dtype=torch.long)
        rewards = torch.tensor(h[0:5,6], dtype=torch.float)

        # 1: Predicted Q values with current state
        pred = self.model(states)
        target = pred.clone()


        #The below will update all the Q scores in the target network for training
        for idx in range(5):
            Q_new = rewards[idx]
            if idx < 4:
                Q_new = rewards[idx] + self.gamma * torch.sum(self.model(states[idx+1]))
            target[idx][actions[idx]-1] = Q_new
            #print(Q_new)
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
