#aqui aniran els models scikit-learn triats
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC

from sklearn.ensemble import HistGradientBoostingClassifier

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



clf1=HistGradientBoostingClassifier(l2_regularization=1.0407479858562003e-05,
                               learning_rate=0.08115041228011703, max_bins=84,
                               max_iter=200, max_leaf_nodes=83,
                               min_samples_leaf=95)
clf2=SVC(C=257.445884056174, class_weight='balanced', gamma=0.18325543501264982,
    kernel='poly', probability=True)