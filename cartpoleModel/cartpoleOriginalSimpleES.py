import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from estorch import ES
import pickle
import statistics
import numpy as np
import random

seed = 7512
torch.manual_seed(seed)
random.seed(seed)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(4, 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, 2)
        self.activation_3 = torch.nn.Sigmoid()

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = self.linear_3(a2)
        a3 = self.activation_3(l3)
        return a3

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
        # torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)

class TrainPacman():
    def __init__(self, startGen=1, device=torch.device('cpu'), std=0.01):
        self.env = gym.make('CartPole-v1')
        self.device = device
        self.std = std

        self.genScoreList = []
        self.bestScores = []
        self.currentGen = startGen
        self.currentRun = 0
        self.allAverages = []

        if startGen==1:
            self.initFirstGen()
            # print("newGen")

    def initFirstGen(self):
        for i in range(100):
            net = Net() # Randomlu Initialize First Generation
            net.apply(init_weights)
            torch.save({'state_dict': net.state_dict()}, f"models/latestGenRun{i}.pt")

    def trainGen(self):
        for i in range(100):
            # print("Gen and Run: ",self.currentGen, self.currentRun)
            self.net = Net()
            model = torch.load(f"models/latestGenRun{self.currentRun}.pt", map_location=torch.device('cpu')) # Load Model
            self.net.load_state_dict(model['state_dict'])

            params=torch.nn.utils.parameters_to_vector(self.net.parameters())
            # print(params[0])

            observation = self.env.reset()
            gameDone = False
            total_reward = 0

            while not gameDone:
                # for _ in range(5):
                observation = (torch.from_numpy(observation)
                               .float()
                               .to(self.device))

                decision = self.net(observation.view(-1,4))
                decision = decision.argmax()
                decision = decision.detach().numpy()


                observation, reward, gameDone, info = self.env.step(decision)
                # self.env.render()

                total_reward += reward

            self.genScoreList.append(total_reward)
            self.currentRun = self.currentRun + 1
        self.allAverages.append(sum(self.genScoreList)/len(self.genScoreList))

    def saveAllAverages(self):
        print(self.allAverages)
        with open(f'allAveragesForOriginalESstd{self.std}.pkl', 'wb') as f:
            pickle.dump(self.allAverages, f)

    def saveScoreList(self):
        with open('latestScores.pkl', 'wb') as f:
            pickle.dump(self.genScoreList, f)

    def findBestRuns(self):
        with open('latestScores.pkl', 'rb') as f:
            self.genScoreList = pickle.load(f)

        print("-----------")
        for i in range(1):
            indxMax = self.genScoreList.index(max(self.genScoreList))
            self.bestScores.append(indxMax)
            print("Run " +str(indxMax)+" with score of " + str(self.genScoreList[indxMax]))
            self.genScoreList[indxMax] = 0

        print("-----------")

    def showGenScore(self):
        with open('latestScores.pkl', 'rb') as f:
            list = pickle.load(f)
        print('Statistics for  GEN: ' + str(self.currentGen))
        print(max(list))
        print(min(list))
        print(sum(list) / len(list))
        print("-----------")

    def generateNextGen(self):
        parameterList = []
        meanParameter = []
        stdevParameter = []
        minParameter = []
        maxParameter = []

        for i in self.bestScores:
            net = Net()
            net.load_state_dict(torch.load(f"models/latestGenRun{i}.pt")['state_dict'])
            params=torch.nn.utils.parameters_to_vector(net.parameters())
            parameterList.append(params)

        print(f"Number of models in bestScores: {len(self.bestScores)}")

        print(len(parameterList[0]))
        print(parameterList[0][0])
        print("---------------")

        for i in range(len(parameterList[0])):
            parallelParamList = []
            for j in range(len(parameterList)):
                parallelParamList.append(float(parameterList[j][i]))
            # stdevParameter.append(statistics.stdev(parallelParamList))
            meanParameter.append(statistics.mean(parallelParamList))

        # print(stdevParameter[0])
        print(meanParameter[0])
        print("---------------")

        paramList = []
        for j in range(len(parameterList[0])):
            param = np.random.normal(meanParameter[j] , self.std, 100)
            paramList.append(param)


        for i in range(len(paramList[0])):
            newParam = []
            for j in range(len(paramList)):
                newParam.append(paramList[j][i])
            torch.nn.utils.vector_to_parameters(torch.FloatTensor(newParam), net.parameters())
            torch.save({'state_dict': net.state_dict()}, f"models/latestGenRun{i}.pt")


        self.currentGen = self.currentGen + 1
        self.currentRun = 0
        self.genScoreList = []
        self.bestScores = []

stdList = [0.01, 0.02, 0.04, 0.06, 0.1]

for tryStd in stdList:
    tp = TrainPacman(std=tryStd)
    for i in range(100):
        tp.trainGen()
        tp.saveScoreList()
        tp.showGenScore()
        tp.findBestRuns()
        tp.generateNextGen()
    tp.saveAllAverages()
