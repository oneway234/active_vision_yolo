import torch
import torch.nn as nn
import agents.dqn as net
import  numpy as np
import agents.img_cnn as fnet

# Hyper Parameters
BATCH_SIZE = 1
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 30
N_ACTIONS = 7
N_STATES = 316368

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = net.DQN(), net.DQN()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # print("actions_value", actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:   # random
            action = np.random.randint(0, N_ACTIONS-1)
        return action

    def store_transition(self, s, a, r, s_):
        s = s.detach().numpy()
        s_ = s_.detach().numpy()
        s = s.ravel()
        s_ = s_.ravel()
        ar = np.append(a, r)
        transition = np.hstack((s, ar, s_))
        # print(transition.shape)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # q_eval w.r.t the action in experience
        b_s = torch.unsqueeze(torch.FloatTensor(b_s), 0)
        b_s = torch.unsqueeze(torch.FloatTensor(b_s), 0)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        b_s_ = torch.unsqueeze(torch.FloatTensor(b_s_), 0)
        b_s_ = torch.unsqueeze(torch.FloatTensor(b_s_), 0)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
