import torch
import torch.nn as nn
import torch.nn.functional as F
from pacmanPygame import playFrame
import pickle
import statistics
import numpy as np
import random
# from estorch import es

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(22, 100)
        self.fc2 = nn.Linear(100, 200)
        # self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(200, 4)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.sig(self.fc4(x))
        return x

# class pacmanAgent():
#     def __init__(self):
#         pass

#
#     def rollout(self, policy, render=False):
#         done = False
#         observation = self.env.reset()
#         total_reward = 0
#         with torch.no_grad():
#             while not done:
#                 observation = (torch.from_numpy(observation).float())
#                 action = policy(observation).max(0)[1].item()
#                 observation, reward, done, info = self.env.step(action)
#                 if render:
#                     self.env.render()
#                 total_reward += reward
#         return total_reward, None
#
#
# tp = TrainPacman()
#
# es = ES(CartPolePolicy, CartPole, torch.optim.Adam, population_size=100)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
        # torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)

class TrainPacman():
    def __init__(self, startGen=1):
        self.genScoreList = []
        self.bestScores = []
        self.currentGen = startGen
        self.currentRun = 0

        if startGen==1:
            self.initFirstGen()

    def initFirstGen(self):
        for i in range(100):
            net = Net() # Randomlu Initialize First Generation
            net.apply(init_weights)
            torch.save({'state_dict': net.state_dict()}, "models/gen"+str(self.currentGen)+"run"+str(i)+".pt")

    def trainGen(self):
        for i in range(100):
            print("Gen and Run: ",self.currentGen, self.currentRun)
            self.net = Net()
            model = torch.load("models/gen"+str(self.currentGen)+"run"+str(self.currentRun)+".pt", map_location=torch.device('cpu')) # Load Model
            self.net.load_state_dict(model['state_dict'])
            params=torch.nn.utils.parameters_to_vector(self.net.parameters())
            print(params[0])

            gameDone = False
            game = playFrame()
            # output = (0, 0, 0, 0)
            decision = 0
            # counter=0
            while not gameDone:
                # for _ in range(5):
                input, score, gameDone, wallDirs = game.main(decision)
                    # if gameDone == True:
                    #     break
                input = torch.FloatTensor(input)
                # print(input)
                output = self.net(input.view(-1,22))

                output = output.squeeze(0)
                output = output / sum(output)
                output[1] = output[0] + output[1]
                output[2] = output[1] + output[2]
                output[3] = output[2] + output[3]
                # print(output)
                chance = random.random()
                if chance > output[2]:
                    decision = 3
                elif chance > output[1]:
                    decision = 2
                elif chance > output[0]:
                    decision = 1
                else:
                    decision = 0
                # output = output.argmax()
                # while(wallDirs[indMax] == 0):
                #     counter = counter + 1
                #     output[indMax] = min(output) - max(output)
                #     indMax = output.argmax()
                #     if counter == 1000:
                #         gameDone = True
                #         break
                # counter = 0

                # for i in range(4):
                #     if (i == indMax):
                #         output.data[i] = 1
                #     else:
                #         output.data[i] = 0

            self.genScoreList.append(score)
            self.currentRun = self.currentRun + 1

    def saveScoreList(self):
        with open('scoreListGen'+str(self.currentGen)+'.pkl', 'wb') as f:
            pickle.dump(self.genScoreList, f)

    def findBestRuns(self):
        with open('scoreListGen'+str(self.currentGen)+'.pkl', 'rb') as f:
            self.genScoreList = pickle.load(f)

        print("-----------")
        for i in range(10):
            indxMax = self.genScoreList.index(max(self.genScoreList))
            self.bestScores.append(indxMax)
            print("Run " +str(indxMax)+" with score of " + str(self.genScoreList[indxMax]))
            self.genScoreList[indxMax] = 0

        print("-----------")

    def showGenScore(self):
        with open('scoreListGen'+str(self.currentGen)+'.pkl', 'rb') as f:
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
            net.load_state_dict(torch.load("models/gen"+str(self.currentGen)+"run"+str(i)+".pt")['state_dict'])
            params=torch.nn.utils.parameters_to_vector(net.parameters())
            parameterList.append(params)

        print(len(parameterList[0]))
        print(parameterList[6][0])
        print("---------------")

        ## Below Part is for uniform distrubition
        # for i in range(len(parameterList[0])):
        #     parallelParamList = []
        #     for j in range(len(parameterList)):
        #         parallelParamList.append(float(parameterList[j][i]))
        #     minParameter.append(max(parallelParamList))
        #     maxParameter.append(min(parallelParamList))
        #
        # print("----------")
        # print(minParameter[0])
        # print(maxParameter[0])
        #
        # paramList = []
        # for j in range(len(parameterList[0])):
        #     param = np.random.uniform(minParameter[j] ,maxParameter[j], 100)
        #     paramList.append(param)

        ## Below part calculates the mean and stdev of every different param through all models
        for i in range(len(parameterList[0])):
            parallelParamList = []
            for j in range(len(parameterList)):
                parallelParamList.append(float(parameterList[j][i]))
            stdevParameter.append(statistics.stdev(parallelParamList))
            meanParameter.append(statistics.mean(parallelParamList))

        print(stdevParameter[0])
        print(meanParameter[0])
        print("---------------")

        # with open('stdevOfBestsGen'+str(self.currentGen)+'.pkl', 'wb') as f:
        #     pickle.dump(stdevParameter, f)
        # with open('meanOfBestsGen'+str(self.currentGen)+'.pkl', 'wb') as f:
        #     pickle.dump(meanParameter, f)

        with open('stdevOfBestsGenLast.pkl', 'wb') as f:
            pickle.dump(stdevParameter, f)
        with open('meanOfBestsGenLast.pkl', 'wb') as f:
            pickle.dump(meanParameter, f)

        paramList = []
        for j in range(len(parameterList[0])):
            param = np.random.normal(meanParameter[j] ,stdevParameter[j], 100)
            paramList.append(param)


        for i in range(len(paramList[0])):
            newParam = []
            for j in range(len(paramList)):
                newParam.append(paramList[j][i])
            torch.nn.utils.vector_to_parameters(torch.FloatTensor(newParam), net.parameters())
            torch.save({'state_dict': net.state_dict()}, "models/gen"+str(self.currentGen+1)+"run"+str(i)+".pt")


        self.currentGen = self.currentGen + 1
        self.currentRun = 0
        self.genScoreList = []
        self.bestScores = []

tp = TrainPacman()
for i in range(50):
    tp.trainGen()
    tp.saveScoreList()
    tp.showGenScore()
    tp.findBestRuns()
    tp.generateNextGen()

# tp = TrainPacman()
# for i in range(10):
#     tp.showGenScore()
#     tp.currentGen = tp.currentGen + 1
