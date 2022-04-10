# Aqui el codi amb totes les proves

import pandas as pd
import torch
# Funcio per a llegir dades en format csv

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset('../data/Obfuscated-MalMem2022.csv')

print("Primer observem les dades y estadístiques sobre les dades")
print(dataset.head())
print(dataset.describe())
data = dataset.values

print("Seleccionem la variable objectiu i mirarem les dimensionalitats de les nostres dades")

print("Dimensionalitat de la BBDD:", dataset.shape)
x = data[:, 0:]
y = data[:, 0]
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

print("Posteriorment mirarem el tipus de dades que tenim")
print(dataset.dtypes)

print("Aixó com també observarem el nombre de dades nules que tenim")
print(dataset.isnull().sum())

with open('../results/estadistiques_dades.txt', 'w') as f:
    f.write(dataset.describe().to_string())