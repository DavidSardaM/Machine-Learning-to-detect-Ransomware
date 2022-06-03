
#----------------------------------------------------------------------------------------------
#                                                                                               #
#              IMPORTING LIBRARIES                                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               CREATING A NEURAL NET                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

class XarxaNeuronal(nn.Module):


    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(43, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 200)
        #self.fc4 = nn.Linear(200, 25)
        self.fc5 = nn.Linear(200, 2)

    def forward(self, x):
        # add dropout layer
        #x = self.dropout(x)
        # add 1st layer (input layer)
        out= self.fc1(x.float())
        out = F.leaky_relu(out)
        # add 1st hidden layer, with relu activation function
        out = F.leaky_relu(self.fc2(out))
        # add 2nd hidden layer, with relu activation function
        out = F.leaky_relu(self.fc3(out))
        # add last layer (outpur layer)
        out = F.leaky_relu(self.fc5(out))

        return out

#----------------------------------------------------------------------------------------------
#                                                                                               #
#           LOADING AND PREPROCESSING DATASET                                                   #
#                                                                                               #
#-----------------------------------------------------------------------------------------------



def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset_predir = load_dataset('../data/Obfuscated-MalMem2022.csv')


dataset_benign = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Benign'))]
print("Dimensionalitat Benignes:", dataset_benign.shape)
dataset_ransom = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Ransomware'))]

print("Dimensionalitat Ransomwares:", dataset_ransom.shape)

dataset_predir = pd.concat([dataset_benign, dataset_ransom]) #we kept only the benign or Ransomware atack samples



dataset_predir['Class'] = dataset_predir['Class'].replace(['Benign','Malware'],[0,1]) #convert to numeric


#delete the no necessary atributes

dataset_predir=dataset_predir.drop(['Category'], axis=1)
dataset_predir=dataset_predir.drop(['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services','callbacks.ngeneric', 'callbacks.nanonymous'], axis=1)


dataset_predir=dataset_predir.drop(['pslist.avg_handlers', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_load_avg',
                                    'malfind.protection', 'psxview.not_in_pslist',  'psxview.not_in_session_false_avg',
                                    'psxview.not_in_csrss_handles'], axis=1)




data = dataset_predir.values

X = data[:,:-1]
y = data[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#NORMALIZE THE X TRAIN AND TEST INDIVIDUALY TO NOT AFFECT THE OTHERS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.fit_transform(X_test)


#----------------------------------------------------------------------------------------------
#                                                                                               #
#               INITIALIZING THE MODEL                                                          #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

model=XarxaNeuronal()  # create the model
nSamples = [len(y[y == 0]), len(y[y == 1])]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights)
optimizer = torch.optim.SGD(list(model.parameters()) + list(normedWeights), lr=0.001, momentum=0.9) #define the optimizator


loss_fn = nn.CrossEntropyLoss(weight=normedWeights.float()) #define the los function


X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               TRAINING THE MODEL                                                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

EPOCHS = 1000 #Number of iterations to train the model


loss_list = np.zeros((EPOCHS,))
accuracy_list_train = np.zeros((EPOCHS,))
accuracy_list_test  = np.zeros(EPOCHS)
n_classes=2
elapsed=0
elapsed_train=0
start_train = time.time()
for epoch in tqdm.trange(EPOCHS):

    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss

    correct = (torch.argmax(y_pred, dim=1) == y_train).type(torch.FloatTensor)
    accuracy_list_train[epoch] = correct.mean()

    # Zero gradients
    optimizer.zero_grad()
    #recalculate with backpropagation
    loss.backward()
    #actualizate the weights
    optimizer.step()
    #if (epoch + 1) % 50 == 0:
    #    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {loss:.4f}, Test Loss: {loss_test.item():.4f}")


    with torch.no_grad():  #Prediction of the model with the test isntances


        start = time.time()

        y_pred = model(X_test)
        y_pred_class=torch.argmax(y_pred, dim=1)
        correct = (y_pred_class == y_test).type(torch.FloatTensor)

        done = time.time()
        elapsed = done - start




        accuracy_list_test[epoch] = correct.mean()
        y_pred_num=np.array(y_pred)

done_train = time.time()
#torch.save(model.state_dict(), "../models/xarxa_neuronal.sav")

#----------------------------------------------------------------------------------------------
#                                                                                               #
#               SHOWING RESULTS OF THE MODEL                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------


print("Ara avaluarem el rendiment de la xarxa neuronal")

print("La taxa encert final de les dades d'entrenament és:", accuracy_list_train[-1])
print("La taxa encert final de les dades de test és", accuracy_list_test[-1])

print('F1: {}'.format(f1_score(y_test.numpy(), y_pred_class.numpy())))
print('Precision: {}'.format(precision_score(y_test.numpy(), y_pred_class.numpy())))
print('Recall: {}'.format(recall_score(y_test.numpy(), y_pred_class.numpy())))
print("Temps en predir:" , elapsed)
elapsed_train = done_train - start_train
print("Temps en entrenar:" , elapsed_train)



# PLOT THE EVOLUTION OF THE ACCURACY

plt.figure(figsize=(10,10))
plt.plot(accuracy_list_train, label='train loss')
plt.plot(accuracy_list_test, label='test loss')
plt.legend()
#plt.savefig("../figures/accuracy_XarxaNeuronal_Ransom_2000.png")
plt.show()

pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability


# PLOT THE ROC CURVE

plt.figure()
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})' ''.format(i, roc_auc[i]))
plt.title("ROC curve for XarxaNeuronal")
plt.legend()
#plt.savefig("../figures/corba_roc_Xarxa_Neuronal_Ransom.png")
plt.show()

# PLOT THE CONFUSION MATRIX

conf_mat = confusion_matrix(y_test, pred)


plt.figure()
sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
#plt.savefig("../figures/confusion_matrix_XarxaNeuronal_Ransom.png")
plt.show()