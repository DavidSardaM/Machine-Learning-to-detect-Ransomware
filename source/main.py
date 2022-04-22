# Aqui el codi amb totes les proves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import model_selection


import numpy as np
import torch
# Funcio per a llegir dades en format csv
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 3)

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset_predir = load_dataset('../data/Obfuscated-MalMem2022.csv')



dataset_benign = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Benign'))]
print("Dimensionalitat Benignes:", dataset_benign.shape)

dataset_ransom = dataset_predir[(dataset_predir['Category'].astype(str).str.startswith('Ransomware'))]

print("Dimensionalitat Ransomwares:", dataset_ransom.shape)





dataset_predir = pd.concat([dataset_benign, dataset_ransom])


dataset_predir=dataset_predir.drop(['Category'], axis=1) #La classe categoria i classe ja no aporten informació diferent no té sentit conservar les dos, les dues es poden identificar com objectiu, no té sentit predir que una instancia es malware sabent que es Ransomware.
dataset_predir['Class'] = dataset_predir['Class'].replace(['Benign','Malware'],[0,1]) #convertim el malware i programes benignes en 0 i 1, es més fàcil treballar amb valors numèrics




print("Primer observem les dades y estadístiques sobre les dades")
print(dataset_predir.head())

print(dataset_predir.describe())
data = dataset_predir.values

print("Seleccionem la variable objectiu i mirarem les dimensionalitats de les nostres dades")

print("Dimensionalitat de la BBDD:", dataset_predir.shape)
x = data[:, :-1]
y = data[:, -1]


print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


print("Posteriorment mirarem el tipus de dades que tenim")
print(dataset_predir.dtypes)


print("Observem nombre valors diferents per cada columna")
for col in dataset_predir.columns:
    print("Nombre valors diferents de l'atribut " + col + ":" + str(len(dataset_predir[col].unique())))


#eliminem les categories que només tenen un possible valor


print("Aixó com també observarem el nombre de dades nules que tenim")
print(dataset_predir.isnull().sum())


with open('../results/estadistiques_dades.txt', 'w') as f:
    f.write(dataset_predir.describe().to_string())




def distribucions(dataset):
    #print("Un cop començades a veure les dades passarem a observar les distribucions i relacions que considerem interessants")
    #print("Primer observarem la matriu de correlació dels atributs")
    #plt.figure()
    #fig, ax = plt.subplots(figsize=(25, 25))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    #plt.title("Matriu de correlació de Pearson")
    #sns.heatmap(dataset.corr(), annot=True, ax=ax, linewidths=.0, annot_kws={"fontsize": 750 / np.sqrt(len(dataset))},
                #square=True)
    #plt.savefig("../figures/pearson_correlation_matrix_.png")
    #plt.show()





    print("Finalment veurem la distribució de la variables objectiu i si les classes es troben balancejades, si és el cas la precisió de les dades serà molt més reprsentativa de les dades")
    plt.figure()
    ax = sns.countplot(x="Class", data=dataset, palette={0: 'thistle', 1: "lightskyblue"})
    plt.suptitle("Target attribute distribution (Class)")
    label = ["benignes", "ransomware"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Classe')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    porc_pot = (len(dataset[dataset.Class == 0]) / len(dataset.Class)) * 100
    print('El percentatge de mostres que son bemignes representa un {:.2f}% del total de dades'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Class == 1]) / len(dataset.Class)) * 100
    print('El percentatge de mostres que son malware representa un {:.2f}% del total de dades'.format(porc_pot))



#distribucions(dataset_predir)
dataset_predir=dataset_predir.drop(['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services'], axis=1)


dataset_predir=dataset_predir.drop(['pslist.avg_handlers', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_load_avg',
                                    'malfind.protection', 'psxview.not_in_pslist',  'psxview.not_in_session_false_avg',
                                    'psxview.not_in_csrss_handles'], axis=1)




def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)


#dataset_predir=standardize_mean(dataset_predir)


data = dataset_predir.values
x = data[:, :-1]
y = data[:, -1]

print("Dimensionalitat de la BBDD:", dataset_predir.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)




# +--------------------------+
# | NO VA PER MEMORIA         |
# +--------------------------+

"""
    #print("Posteriorment passarem a veure els histogrames de les variables")
    #print("Primer els histogrames de tots els atributs")
    #plt.figure()
    #sns.pairplot(dataset)
    #plt.savefig("../figures/histograma.png")
    #plt.show()
    
    
    
    print("Després els histogrames dels atributs segons la classe objectiu")

    plt.figure()
    sns.pairplot(dataset, hue="Class", palette={0: 'thistle', 1: "lightskyblue"})
    plt.savefig("../figures/histograma_per_classes.png")
    plt.show()
"""

models = []

models.append(('SVM rbf gamma 0.7', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='rbf', gamma=0.7, probability=True))))
models.append(('SVM sigmoide gamma 0.7', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='sigmoid', gamma=0.7, probability=True))))
models.append(('SVM precomputed gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='precomputed', gamma=0.7, probability=True))))
models.append(('SVM polinomi gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='poly', gamma=0.7, probability=True))))
models.append(('SVM linear gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='linear', gamma=0.7, probability=True))))
models.append (('Logistic Regression', make_pipeline(MinMaxScaler(),LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', max_iter=200000))))
models.append (('Guassian Naive Bayes', make_pipeline(MinMaxScaler(),GaussianNB())))
models.append (('Linear Discriminant Analysis', make_pipeline(MinMaxScaler(),LinearDiscriminantAnalysis())))
models.append (('Decision Tree', make_pipeline(MinMaxScaler(),DecisionTreeClassifier())))
models.append (('K Nearest Neigbors', make_pipeline(MinMaxScaler(),KNeighborsClassifier())))
models.append (('Extra Trees', make_pipeline(MinMaxScaler(),ExtraTreesClassifier(n_estimators=100))))
models.append (('Random Forest',  make_pipeline(MinMaxScaler(),RandomForestClassifier( n_estimators=150, n_jobs=-1))))
models.append (('HistGradientBoosting', make_pipeline(MinMaxScaler(),HistGradientBoostingClassifier(max_iter=100))))
models.append (('ADABoosting', make_pipeline(MinMaxScaler(),AdaBoostClassifier(n_estimators=150))))
models.append (('Bagging Classifier', make_pipeline(MinMaxScaler(),BaggingClassifier( GaussianNB(), max_samples=0.9, max_features=0.9))))
models.append (('Perceptró', make_pipeline(MinMaxScaler(),Perceptron(fit_intercept=False, max_iter=100, shuffle=True))))
models.append (('GradientBoostingClassifier', make_pipeline(MinMaxScaler(),GradientBoostingClassifier(n_estimators=50))))





scoring = ['balanced_accuracy','f1_weighted',  'recall_weighted',  'roc_auc_ovr_weighted']



for index, (name, model) in enumerate(models):
    K_Fold = model_selection.KFold (n_splits = 6, shuffle=True)
    cv_results = model_selection.cross_validate (model, x, y, cv = K_Fold, scoring = scoring)
    message =  "%s  dades (%f):  accuracy: %f (%f),  f1: %f, recall: %f, roc: %f tiempo %f " % (name, 6,cv_results['test_balanced_accuracy'].mean(),
                                    cv_results['test_balanced_accuracy'].std(),  cv_results['test_f1_weighted'].mean(), cv_results['test_recall_weighted'].mean(),
                                    cv_results['test_roc_auc_ovr_weighted'].mean(), cv_results['fit_time'].mean() )
    print (message)