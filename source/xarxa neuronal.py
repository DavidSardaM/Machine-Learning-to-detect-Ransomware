#----------------------------------------------------------------------------------------------
#                                                                                               #
#              IMPORTING LIBRARIES                                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               CREATING A NEURAL NET                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

class XarxaNeuronal(nn.Module):


    def __init__(self, in_size, out_size):
        super().__init__()  # Call the init function of nn.Module
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 25)
        self.fc4 = nn.Linear(25, out_size)

    def forward(self, x):

        # add 1st hidden layer, with relu activation function
        out= self.fc1(x.float())
        out = F.leaky_relu(out)

        # add 2nd hidden layer, with relu activation function
        out = F.leaky_relu(self.fc2(out))

        # add dropout layer
        out = self.dropout(out)

        # add 3rd hidden layer, with relu activation function
        out = F.leaky_relu(self.fc3(out))

        # add output layer, with relu activation function
        out = F.leaky_relu(self.fc4(out))

        return out


#----------------------------------------------------------------------------------------------
#                                                                                               #
#               IMPORTING THE DATASET IRIS                                                      #
#                                                                                               #
#-----------------------------------------------------------------------------------------------



iris = datasets.load_iris()

X = iris['data']
y = iris['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #NORMALIZE X

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               DEFINING THE NEURAL NET                                                         #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

model=XarxaNeuronal(4,3)


nSamples = [len(y[y == 0]), len(y[y == 1]), len(y[y == 2])]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
#device="cpu"  #IT CAN BE CONFIGURABLE WITH CUDA
normedWeights = torch.FloatTensor(normedWeights)#.to(device)
loss_fn = nn.CrossEntropyLoss(weight=normedWeights.float())
names = iris['target_names']
feature_names = iris['feature_names']

optimizer = torch.optim.SGD(list(model.parameters()) + list(normedWeights), lr=0.001, momentum=0.9)

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               ANALIZING THE DATASET                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    ax1.plot(X_plot[:, 0], X_plot[:, 1],
             linestyle='none',
             marker='o',
             label=target_name)
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis('equal')
ax1.legend();

for target, target_name in enumerate(names):
    X_plot = X[y == target]
    ax2.plot(X_plot[:, 2], X_plot[:, 3],
             linestyle='none',
             marker='o',
             label=target_name)
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis('equal')
ax2.legend();

plt.savefig("../figures/analisi_datasetiris.png")


#----------------------------------------------------------------------------------------------
#                                                                                               #
#               TRAINING THE MODEL                                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

EPOCHS = 1000
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

loss_list = np.zeros((EPOCHS,))
accuracy_list_train = np.zeros((EPOCHS,))
accuracy_list_test  = np.zeros(EPOCHS)
n_classes=3
for epoch in tqdm.trange(EPOCHS):
    #CALCULATE THE PREDICTION WITH THE CURRENT STATE OF THE MODEL
    y_pred = model.forward(X_train)
    #COMPUTE THE LOST FUNCTION
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss

    correct = (torch.argmax(y_pred, dim=1) == y_train).type(torch.FloatTensor)
    accuracy_list_train[epoch] = correct.mean()

    # Zero gradients
    optimizer.zero_grad()


    #RECALCULATE WEIGHTS WITH BACKPROPAGATION
    loss.backward()
    #ACTUALIZATE WEIGHTS
    optimizer.step()



    with torch.no_grad(): # PREDICTION OF THE TEST SAMPLES
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list_test[epoch] = correct.mean()
        y_pred_num=np.array(y_pred)



#----------------------------------------------------------------------------------------------
#                                                                                               #
#              PLOTING RESULTS                                                                  #
#                                                                                               #
#-----------------------------------------------------------------------------------------------



# PLOT THE ROC CURVE

plt.figure()
fpr = {}
tpr = {}
roc_auc = {}
print(y_pred_num.shape)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_num[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(y_pred_num.shape)
print(y.shape)
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.title("ROC curve for XarxaNeuronal")
plt.legend()
plt.savefig("../figures/corba_roc_Xarxa_Neuronal_iris.png")
plt.show()


print("La precisió final de les dades d'entrenament és:", accuracy_list_train[-1])
print("La precisió final de les dades de test és", accuracy_list_test[-1])


# PLOT THE EVOLUTION OF THE ACCURACY ON THE TRAINING ITERATIONS

plt.figure(figsize=(10,10))
plt.plot(accuracy_list_train, label='train loss')
plt.plot(accuracy_list_test, label='test loss')
plt.legend()
plt.savefig("../figures/accuracy_XArxaNeuronal_iris.png")
plt.show()