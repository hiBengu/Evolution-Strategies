import gym
import torch
from estorch import ES
from pacmanPygame import playFrame
import random
import numpy as np

SEED = 583
# Reproducibility
# random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Policy(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Policy, self).__init__()
        self.linear_1 = torch.nn.Linear(n_input, 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, n_output)
        # self.activation_3 = torch.nn.ReLU()

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = self.linear_3(a2)
        # a3 = self.activation_3(l3)
        return l3

class Agent():
    def __init__(self, device=torch.device('cpu')):
        self.env = playFrame()
        self.device = device


    def rollout(self, policy, render=False):
        global totReward, counter
        done = False
        observation = self.env.resetGame()
        total_reward = 0
        with torch.no_grad():
            while not done:
                observation = (torch.from_numpy(observation)
                               .float()
                               .to(self.device))
                output = policy(observation)
                output = output.squeeze(0)

                output = output / sum(output)
                output[1] = output[0] + output[1]
                output[2] = output[1] + output[2]
                output[3] = output[2] + output[3]

                chance = random.random()
                if chance > output[2]:
                    decision = 3
                elif chance > output[1]:
                    decision = 2
                elif chance > output[0]:
                    decision = 1
                else:
                    decision = 0

                for _ in range(5):
                    observation, reward, done, wallDirs = self.env.main(decision)
                    if done:
                        break
                if render:
                    self.env.render()
                total_reward = reward
        totReward += total_reward
        counter += 1
        if counter == 100:
            print("Average Reward: " + str(totReward/100))
            counter = 0
            totReward = 0
        return total_reward

if __name__ == '__main__':
    global counter, totReward
    counter = 0
    totReward = 0
    device = torch.device("cpu")
    agent = Agent()
    n_input = agent.env.observation_space.shape[0]
    n_output = agent.env.action_space.shape[0]
    print(n_input, n_output)
    es = ES(Policy, Agent, torch.optim.Adam, population_size=100, sigma=0.02,
            device=device, policy_kwargs={'n_input': n_input, 'n_output': n_output},
            agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.01})
    es.train(n_steps=100, n_proc=1)

    # Latest Policy
    reward = agent.rollout(es.policy, render=False)
    print(f'Latest Policy Reward: {reward}')

    # Policy with the highest reward
    policy = Policy(n_input, n_output).to(device)
    policy.load_state_dict(es.best_policy_dict)
    reward = agent.rollout(policy, render=False)
    print(f'Best Policy Reward: {reward}')
    torch.save({'state_dict': policy.state_dict()}, "bestES.pt")
