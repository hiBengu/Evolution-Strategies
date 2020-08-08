from matplotlib import pyplot as plt
import pickle
import numpy as np
data = []

for i in range(1,50):
    with open('scoreListGen'+str(i)+'.pkl', 'rb') as f:
        list = pickle.load(f)
    avgScore = sum(list)/len(list)
    # if i > 20:
    #     avgScore = 1.6 * avgScore
    data.append(avgScore)

plt.xlabel('Generation', fontsize=18)
plt.ylabel('Average Score', fontsize=16)
plt.plot(data) # plotting by columns

# file = "allAverages1Hidden64allRelu.pkl"
#
# with open(file, 'rb') as f:
#     data = pickle.load(f, encoding='bytes')
#
# plt.plot(data, label="1 Hidden Layer") # plotting by columns

plt.legend()
plt.show()
