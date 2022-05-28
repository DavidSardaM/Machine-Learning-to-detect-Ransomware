

from source.modelsfinals import XarxaNeuronal
import torch
import pandas as pd
from torch.autograd import Variable
import numpy as np
import pickle


#----------------------------------------------------------------------------------------------
#                                                                                               #
#             CARREGAR LES DADES A PREDIR                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

# Cal modificar amb les dades a predir, tenint en compte eliminar les categories eliminades per la predicció



def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset_predir = load_dataset('../data/Obfuscated-MalMem2022.csv')
dataset_benign = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Benign'))]
dataset_ransom = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Ransomware'))]
dataset_predir = pd.concat([dataset_benign, dataset_ransom])

# ELIMINEM CATEGORIES INNECESSÀRIES

dataset_predir=dataset_predir.drop(['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services','callbacks.ngeneric', 'callbacks.nanonymous',
                                    'pslist.avg_handlers', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_load_avg',
                                    'malfind.protection', 'psxview.not_in_pslist',  'psxview.not_in_session_false_avg',
                                    'psxview.not_in_csrss_handles', 'Category'], axis=1)

dataset_predir['Class'] = dataset_predir['Class'].replace(['Benign','Malware'],[0,1])

# CARREGUEM LES DADES A PREDIR
demo=dataset_predir.sample(frac = 0.1)
#----------------------------------------------------------------------------------------------
#                                                                                               #
#              PREDICCIÓ AMB ELS MODELS DEL GITHUB                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------

#DESCOMENTAR PER PREDIR AMB MODEL XARXA NEURONAL PYTORCH

"""
model = XarxaNeuronal()
model.load_state_dict(torch.load('../models/xarxa_neuronal.sav'))
model.eval()
demo_x = Variable(torch.from_numpy(demo.values[:,:-1])).float()

y_pred_prob = model(demo_x)
y_pred = torch.argmax(y_pred_prob, dim=1)
"""


# Descomentar per executar amb model HistGradientBoosting
#loaded_model = pickle.load(open('../models/modelHistGradientBoosting_optimitzat.sav', 'rb'))

# Descomentar per executar amb model SVM
#loaded_model = pickle.load(open('../models/modelSVM_optimitzat', 'rb'))


#DESCOMENTAR PER PREDIR AMB MODEL SKLEARN (HISTGRADIENTBOOSTING O SVM)
"""
demo_x=demo.values[:,:-1]
demo_y=demo.values[:,-1]
y_pred = loaded_model.predict(demo_x)

"""
# MOSTRA RESULTAT PREDICCIÓ
print(y_pred)
