import torch
import torch.nn as nn
import torch.nn.functional as F
from pacmanPygame import playFrame
import pickle
import statistics
import numpy as np
# from estorch import es

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(398, 100)
        # self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        torch.nn.init.normal_(m.weight, mean=0.0, std=5.0)
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
            output = (0, 1, 0, 0)
            while not gameDone:
                score, input, gameDone = game.main(output)

                output = self.net(input.view(-1,398))
                indMax = output.argmax()
                output = output.squeeze(0)
                for i in range(4):
                    if (i == indMax):
                        output.data[i] = 1
                    else:
                        output.data[i] = 0

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
        print("-----------")
        print('Statistics')
        print(max(list))
        print(min(list))
        print(sum(list) / len(list))
        print("-----------")

    def generateNextGen(self):
        parameterList = []
        meanParameter = []
        stdevParameter = []

        for i in self.bestScores:
            net = Net()
            net.load_state_dict(torch.load("models/gen"+str(self.currentGen)+"run"+str(i)+".pt")['state_dict'])
            params=torch.nn.utils.parameters_to_vector(net.parameters())
            parameterList.append(params)

        print(len(parameterList[0]))
        print(parameterList[6][0])
        print("---------------")

        for i in range(len(parameterList[0])):
            parallelParamList = []
            for j in range(len(parameterList)):
                parallelParamList.append(float(parameterList[j][i]))
            stdevParameter.append(statistics.stdev(parallelParamList))
            meanParameter.append(statistics.mean(parallelParamList))

        print(stdevParameter[0])
        print(meanParameter[0])
        print("---------------")

        with open('stdevOfBestsGen'+str(self.currentGen)+'.pkl', 'wb') as f:
            pickle.dump(stdevParameter, f)
        with open('meanOfBestsGen'+str(self.currentGen)+'.pkl', 'wb') as f:
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
for i in range(10):
    tp.trainGen()
    tp.saveScoreList()
    tp.showGenScore()
    tp.findBestRuns()
    tp.generateNextGen()
