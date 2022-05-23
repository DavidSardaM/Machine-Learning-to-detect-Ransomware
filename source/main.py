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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc, accuracy_score, make_scorer

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
    print(col)
    print("Nombre valors diferents de l'atribut " + col + ":" + str(len(dataset_predir[col].unique())))


#eliminem les categories que només tenen un possible valor


print("Aixó com també observarem el nombre de dades nules que tenim")
print(dataset_predir.isnull().sum())


with open('../results/estadistiques_dades.txt', 'w') as f:
    f.write(dataset_predir.describe().to_string())




def distribucions(dataset):
    print("Un cop començades a veure les dades passarem a observar les distribucions i relacions que considerem interessants")
    print("Primer observarem la matriu de correlació dels atributs")
    plt.figure()
    fig, ax = plt.subplots(figsize=(25, 25))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Matriu de correlació de Pearson")
    sns.heatmap(dataset.corr(), annot=True, ax=ax, linewidths=.0, annot_kws={"fontsize": 750 / np.sqrt(len(dataset))}, square=True)
    plt.savefig("../figures/pearson_correlation_matrix.png")
    plt.show()



"""

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


"""
#distribucions(dataset_predir)
dataset_predir=dataset_predir.drop(['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services','callbacks.ngeneric', 'callbacks.nanonymous'], axis=1)


dataset_predir=dataset_predir.drop(['pslist.avg_handlers', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_load_avg',
                                    'malfind.protection', 'psxview.not_in_pslist',  'psxview.not_in_session_false_avg',
                                    'psxview.not_in_csrss_handles'], axis=1)




def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)


data=standardize_mean(dataset_predir)



x = data[:, :-1]
y = data[:, -1]


print(y)

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
    message =  "%s,  dades (%f):  accuracy: %f (%f),  f1: %f, recall: %f, roc: %f, temps convergir algoritme %f, i temps test: %f " % (name, 6,cv_results['test_balanced_accuracy'].mean(),
                                    cv_results['test_balanced_accuracy'].std(),  cv_results['test_f1_weighted'].mean(), cv_results['test_recall_weighted'].mean(),
                                    cv_results['test_roc_auc_ovr_weighted'].mean(), cv_results['fit_time'].mean(),  cv_results['score_time'].mean())
    print (message)



"""
"""
param = {
    'l2_regularization': loguniform(1e-6, 1e3),
    'learning_rate': loguniform(0.001, 10),
    'max_leaf_nodes': range(2, 256),
    'min_samples_leaf': range(1, 100),
    'max_bins': range(2, 255),
    'max_iter': [10,50,100,200, 500,1000]
}



hgb = HistGradientBoostingClassifier()# Instantiate the grid search model
random_search_extra = RandomizedSearchCV(estimator = hgb, param_distributions = param, cv = 6, n_jobs = -1, n_iter=1000)
random_search_extra = random_search_extra.fit(x, y)
best_estimator1 = random_search_extra.best_estimator_
print(best_estimator1)

param={'C': loguniform(1e0, 1e5),
 'gamma': loguniform(1e-4, 1),
 'kernel': ['poly'],
 'class_weight':['balanced', None]}



svm = SVC( probability=True)# Instantiate the grid search model
random_search_extra = RandomizedSearchCV(estimator = svm, param_distributions = param, cv = 6, n_jobs = -1, n_iter=500)
random_search_extra = random_search_extra.fit(x, y)
best_estimator1 = random_search_extra.best_estimator_
print(best_estimator1)
"""
"""
HistGradientBoostingClassifier(l2_regularization=1.0407479858562003e-05,
                               learning_rate=0.08115041228011703, max_bins=84,
                               max_iter=200, max_leaf_nodes=83,
                               min_samples_leaf=95)
                               
SVC(C=257.445884056174, class_weight='balanced', gamma=0.18325543501264982,
    kernel='poly', probability=True)

"""
models = []

models.append(('HistGradientBoosting optimitzat', make_pipeline(MinMaxScaler(), HistGradientBoostingClassifier(l2_regularization=1.0407479858562003e-05,
                               learning_rate=0.08115041228011703, max_bins=84,
                               max_iter=200, max_leaf_nodes=83,
                               min_samples_leaf=95))))
models.append(('SVM polinomi optimitzat', make_pipeline(MinMaxScaler(), SVC(C=257.445884056174, class_weight='balanced', gamma=0.18325543501264982,
    kernel='poly', probability=True))))

scoring = ['balanced_accuracy','f1_weighted',  'recall_weighted',  'roc_auc_ovr_weighted']



for index, (name, model) in enumerate(models):
    K_Fold = model_selection.KFold (n_splits = 6, shuffle=True)
    cv_results = model_selection.cross_validate (model, x, y, cv = K_Fold, scoring = scoring)
    message =  "%s,  dades (%f):  accuracy: %f (%f),  f1: %f, recall: %f, roc: %f, temps convergir algoritme %f, i temps test: %f " % (name, 6,cv_results['test_balanced_accuracy'].mean(),
                                    cv_results['test_balanced_accuracy'].std(),  cv_results['test_f1_weighted'].mean(), cv_results['test_recall_weighted'].mean(),
                                    cv_results['test_roc_auc_ovr_weighted'].mean(), cv_results['fit_time'].mean(),  cv_results['score_time'].mean())
    print (message)


clf1=HistGradientBoostingClassifier(l2_regularization=1.0407479858562003e-05,
                               learning_rate=0.08115041228011703, max_bins=84,
                               max_iter=200, max_leaf_nodes=83,
                               min_samples_leaf=95)
clf2=SVC(C=257.445884056174, class_weight='balanced', gamma=0.18325543501264982,
    kernel='poly', probability=True)


probs_hgb = cross_val_predict(clf1, x, y, cv=6, method='predict_proba')
"""
y_pred1 = cross_val_predict(clf1, x, y, cv=6)
conf_mat = confusion_matrix(y, y_pred1)
print(conf_mat)
plt.figure()
sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
plt.savefig("../figures/confusion_matrix_HGB.png")
plt.show()
"""

probs_svm = cross_val_predict(clf2, x, y, cv=6, method='predict_proba')

y_pred2 = cross_val_predict(clf2, x, y, cv=6)
conf_mat = confusion_matrix(y, y_pred2)
print(conf_mat)
plt.figure()
sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
plt.savefig("../figures/confusion_matrix_svm.png")
plt.show()







n_classes=2

# Corba precision recall

def analisi_res(probs,y,name):
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y== i, probs[:, i])
        average_precision[i] = average_precision_score(y == i, probs[:, i])

        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, round(average_precision[i])))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
    plt.title("Precision-recall curve for {} model".format(name))
    plt.savefig("../figures/corba_precision_recall_{}.png".format(name))
    plt.show()
    #corba ROC
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y== i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.title("ROC curve for {} model".format(name))
    plt.legend()
    plt.savefig("../figures/corba_roc_{}.png".format(name))
    plt.show()


analisi_res(probs_hgb,y,'HistGadientBoosting')
analisi_res(probs_svm,y,'SVM')