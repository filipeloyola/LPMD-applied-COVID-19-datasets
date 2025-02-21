# My utils
from mypipeline import *


# Libraries
import numpy as np
import pandas as pd
import random as rd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
import os
import time
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

# MICE, KNN, Dumb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

# dataset1
original_dataset = pd.read_excel("HCAI-INFECTION.xlsx")
original_dataset = original_dataset.drop(columns='Unnamed: 0')
original_dataset.columns = ['ID_PACIENTE', 'Idade', 'ALT (TGP)', 'Basófilos', 'Bilirrubina Direta',
       'Bilirrubina Indireta', 'CHCM', 'CK', 'Calcio Ionizavel', 'Creatinina',
       'DHL', 'Dimeros D, quant', 'Eosinófilos', 'Eritrócitos, urina',
       'Fibrinogenio', 'Fosfatase Alcalina', 'Gama-GT', 'Glicose', 'HCM',
       'HCO3 venoso', 'Hemoglobina', 'Leucócitos', 'Leucócitos, urina',
       'Linfócitos', 'Magnésio', 'Monócitos', 'Neutrófilos', 'Plaquetas',
       'Potássio', 'Proteína C-Reativa', 'RDW', 'Sódio', 'TP_INR',
       'TTPA - Paciente_Normal', 'Uréia', 'VCM', 'Volume plaquetário médio',
       'SEXO', 'TARGET', 'Infecção Hospitalar']
dataset1 = original_dataset.copy(deep=True)
print(dataset1.shape)

# dataset2
dataset2 = pd.read_csv("COVID-PROGNOSTIC.csv")
dataset2 = dataset2.drop(columns="Unnamed: 0")
dataset2.columns = ['ID_PACIENTE', 'Idade', 'ALT (TGP)', 'AST (TGO)', 'Basófilos',
       'Basófilos (%)', 'CHCM', 'Creatinina', 'Eosinófilos', 'Eosinófilos (%)',
       'Eritrócitos', 'HCM', 'Hematócrito', 'Hemoglobina', 'Leucócitos',
       'Linfócitos', 'Linfócitos (%)', 'Monócitos', 'Monócitos (%)',
       'Neutrófilos', 'Neutrófilos (%)', 'Plaquetas', 'Potássio',
       'Proteína C-Reativa', 'RDW', 'Sódio', 'Uréia', 'VCM',
       'Volume plaquetário médio', 'SEXO', 'TARGET']
print(dataset2.shape)

# dataset3
dataset3 = pd.read_csv("HSL.csv")
dataset3.columns = ['sex', 'age', 'creatinine', 'creatine phosphokinase', 'd-dimer',
       'eosinophils (%)', 'hemoglobin', 'leukocytes', 'lymphocytes (%)',
       'monocytes (%)', 'neutrophils (%)', 'platelets', 'potassium',
       'c-reactive protein', 'sodium', 'AST', 'ALT', 'troponin', 'urea',
       'TARGET']
print(dataset3.shape)

# dataset4
dataset4 = pd.read_csv("HBP.csv")
dataset4.columns = ['sex', 'age', 'creatinine', 'creatine phosphokinase', 'd-dimer',
       'eosinophils (%)', 'hemoglobin', 'leukocytes', 'lymphocytes (%)',
       'monocytes (%)', 'neutrophils (%)', 'platelets', 'potassium',
       'c-reactive protein', 'sodium', 'AST', 'ALT', 'troponin', 'urea',
       'TARGET']

print(dataset4.shape)





################################################################################################
# MACHINE LEARNING
################################################################################################

import time

# Dicionário para armazenar tempos de processamento
processing_times = []

#dataset
hosp1 = dataset1
hosp2 = dataset2
hosp3 = dataset3
hosp4 = dataset4

data = {
    "hosp1": ( (hosp1.drop(columns=["Infecção Hospitalar", "TARGET", "ID_PACIENTE"])), hosp1.TARGET),
    "hosp2": ( (hosp2.drop(columns=["TARGET", "ID_PACIENTE"])), hosp2.TARGET),
    "hosp3": ( (hosp3.drop(columns=["TARGET"])), hosp3.TARGET),
    "hosp4": ( (hosp4.drop(columns=["TARGET"])), hosp4.TARGET)
}

#algorithms
algorithms = {
    "SVM": (SVC(probability= True), {"C": [1, 10], "kernel": ("linear", "rbf"), "gamma": ('scale', 'auto')}),
    "RF" : (RandomForestClassifier(random_state=0), {"n_estimators": [100,200,500], "max_depth": [4,6,10],"max_features":[10,20,30]}),
    "GB" : (GradientBoostingClassifier(random_state=0), {"n_estimators": [100,200,500],  'learning_rate': [0.05, 0.1], "max_depth": [4,6,10]})
}

# Definição de imputers para testar diferentes abordagens
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5, weights='distance'),
    "MICE": IterativeImputer(max_iter=100),
    "SAEI": None,
    "PMIVAE": None,  # Será tratado separadamente, pois é um modelo treinado
    "LPMD": None,
    "LPMD2": None  # Será tratado separadamente, pois é um modelo treinado
}

#3 folds to choose the best hyperparameters
gskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=20) 

#choose of the best hyperparameters through balanced accuracy
perf = balanced_accuracy_score

#5-fold cross validation 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20) 

#define Standard Scaler to standardize the features
prep = StandardScaler()

#undersampling the majority class when classes are umbalanced
under = RandomUnderSampler(sampling_strategy='majority', random_state = 0)

ini = time.time()

# Loop sobre cada dataset
for name, (X, y) in data.items():  
    for imputer_name, imputer in imputers.items():  # Loop pelos imputers

        start_time = time.time()  # Início da contagem de tempo
        
        # Dicionários para armazenar os resultados por algoritmo
        score = {alg: [] for alg in algorithms.keys()}
        auc_score = {alg: [] for alg in algorithms.keys()}
    
        # Loop pelos algoritmos
        for algorithm, (clf, parameters) in algorithms.items():
            
            best = GridSearchCV(clf, parameters, cv=gskf, scoring=(make_scorer(perf)))

            for train, test in kf.split(X, y):
                
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]

                # Modelo normalização
                scaler = MinMaxScaler(feature_range=(0, 1))
                model_norm = scaler.fit(X_train)

                # Normaliza X_train                
                X_train_norm = model_norm.transform(X_train)
                X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)

                # Normaliza X_Test
                X_test_norm = model_norm.transform(X_test)
                X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)
                
                y_ = pd.DataFrame.from_dict(y)
               
                if (((y_[y_.TARGET == 1].shape[0]) * 1.5) < (y_[y_.TARGET == 0].shape[0])):
                    
                    y_pred, y_true = [], []
                    
                    X_train, y_train = under.fit_resample(X_train, y_train)

                    #print("Tamanho X_train: ", X_train.shape)

                    #print("Total de NaNs em X_train:", np.isnan(X_train.iloc[:, :].values).sum())

                    #print("Tamanho X_test: ", X_test.shape)
                    
                    #print("Total de NaNs em X_test:", np.isnan(X_test.iloc[:, :].values).sum())
                    
                    # Aplicação do imputer atual
                    if imputer_name == "PMIVAE":
            
                        # Copia os valores originais e cria máscara
                        df_copy_X_train = X_train.copy(deep=True)
                        mask_train = df_copy_X_train.isna()
                        df_copy_X_test = X_test.copy(deep=True)
                        mask_test = df_copy_X_test.isna()

                        # Treinar e imputar com PMIVAE
                        pmivae_model = MyPipeline.model_autoencoder_pmivae(X_train.iloc[:, :].values)
                        imputed_X_train = pmivae_model.transform(X_train.iloc[:, :].values)  
                        imputed_X_test = pmivae_model.transform(X_test.iloc[:,:].values)  

                        #print("Total de NaNs antes da restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs antes da restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())

                        # Transforma em DataFrame para facilitar a manipulação
                        imputed_X_train = pd.DataFrame(imputed_X_train, columns=df_copy_X_train.columns, index=df_copy_X_train.index)
                        imputed_X_test = pd.DataFrame(imputed_X_test, columns=df_copy_X_test.columns, index=df_copy_X_test.index)

                        # Restaura os valores originais, mantendo apenas os imputados
                        imputed_X_train = df_copy_X_train.where(~mask_train, imputed_X_train)
                        imputed_X_test = df_copy_X_test.where(~mask_test, imputed_X_test)

                        # Transforma novamente em array para manter compatibilidade
                        imputed_X_train = imputed_X_train.to_numpy()
                        imputed_X_test = imputed_X_test.to_numpy()

                        #print("Total de NaNs após restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs após restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())


                    elif imputer_name == "SAEI":

                        # Copia os valores originais e cria máscara
                        df_copy_X_train = X_train.copy(deep=True)
                        mask_train = df_copy_X_train.isna()
                        df_copy_X_test = X_test.copy(deep=True)
                        mask_test = df_copy_X_test.isna()

                        # SAEI
                        features = X_train.columns[X_train.isna().any()].tolist()
                        model = MyPipeline.model_saei(  dataset_train_md = X_train,
                                                        dataset_test_md = X_test,
                                                        col_name =  features,
                                                        input_shape = X.shape[1])
                        
                        # Imputação dos missing values nos conjuntos de treino e teste
                        imputed_X_train = model.transform(X_train.iloc[:, :].values)  
                        imputed_X_test = model.transform(X_test.iloc[:,:].values) 
                        
                        #print("Total de NaNs antes da restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs antes da restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())

                        # Transforma em DataFrame para facilitar a manipulação
                        imputed_X_train = pd.DataFrame(imputed_X_train, columns=df_copy_X_train.columns, index=df_copy_X_train.index)
                        imputed_X_test = pd.DataFrame(imputed_X_test, columns=df_copy_X_test.columns, index=df_copy_X_test.index)

                        # Restaura os valores originais, mantendo apenas os imputados
                        imputed_X_train = df_copy_X_train.where(~mask_train, imputed_X_train)
                        imputed_X_test = df_copy_X_test.where(~mask_test, imputed_X_test)

                        # Transforma novamente em array para manter compatibilidade
                        imputed_X_train = imputed_X_train.to_numpy()
                        imputed_X_test = imputed_X_test.to_numpy()

                        #print("Total de NaNs após restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs após restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())
                    
                    elif imputer_name == "LPMD2":
                        # Treinar e imputar
                        lpmd2_model = MyPipeline.model_lpmd2(X_train, y_train)  # Treina o modelo
                        imputed_X_train = lpmd2_model.transform(X_train, is_Train=True)  # Imputa os dados de treino
                        imputed_X_test = lpmd2_model.transform(X_test)  # Imputa os dados de teste

                    elif imputer_name == "LPMD":
                        # Treinar e imputar
                        lpmd_model = MyPipeline.model_lpmd(X_train, y_train)  # Treina o modelo
                        imputed_X_train = lpmd_model.transform(X_train)  # Imputa os dados de treino
                        imputed_X_test = lpmd_model.transform(X_test)  # Imputa os dados de teste
                    
                    else:
                        imputer.fit(X_train)
                        imputed_X_train = imputer.transform(X_train)
                        imputed_X_test = imputer.transform(X_test)



                else:
                    y_pred, y_true = [], []
                    
                    # Aplicação do imputer atual
                    if imputer_name == "PMIVAE":

                        # Copia os valores originais e cria máscara
                        df_copy_X_train = X_train.copy(deep=True)
                        mask_train = df_copy_X_train.isna()
                        df_copy_X_test = X_test.copy(deep=True)
                        mask_test = df_copy_X_test.isna()

                        # Treinar e imputar com PMIVAE
                        pmivae_model = MyPipeline.model_autoencoder_pmivae(X_train.iloc[:, :].values)
                        imputed_X_train = pmivae_model.transform(X_train.iloc[:, :].values)  
                        imputed_X_test = pmivae_model.transform(X_test.iloc[:,:].values)  

                        #print("Total de NaNs antes da restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs antes da restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())

                        # Transforma em DataFrame para facilitar a manipulação
                        imputed_X_train = pd.DataFrame(imputed_X_train, columns=df_copy_X_train.columns, index=df_copy_X_train.index)
                        imputed_X_test = pd.DataFrame(imputed_X_test, columns=df_copy_X_test.columns, index=df_copy_X_test.index)

                        # Restaura os valores originais, mantendo apenas os imputados
                        imputed_X_train = df_copy_X_train.where(~mask_train, imputed_X_train)
                        imputed_X_test = df_copy_X_test.where(~mask_test, imputed_X_test)

                        # Transforma novamente em array para manter compatibilidade
                        imputed_X_train = imputed_X_train.to_numpy()
                        imputed_X_test = imputed_X_test.to_numpy()

                        #print("Total de NaNs após restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs após restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())

                    elif imputer_name == "SAEI":

                        # Copia os valores originais e cria máscara
                        df_copy_X_train = X_train.copy(deep=True)
                        mask_train = df_copy_X_train.isna()
                        df_copy_X_test = X_test.copy(deep=True)
                        mask_test = df_copy_X_test.isna()

                        # SAEI
                        features = X_train.columns[X_train.isna().any()].tolist()
                        model = MyPipeline.model_saei(  dataset_train_md = X_train,
                                                        dataset_test_md = X_test,
                                                        col_name =  features,
                                                        input_shape = X.shape[1])
                        
                        # Imputação dos missing values nos conjuntos de treino e teste
                        imputed_X_train = model.transform(X_train.iloc[:, :].values)  
                        imputed_X_test = model.transform(X_test.iloc[:,:].values) 
                        
                        #print("Total de NaNs antes da restauração em imputed_X_train:", np.isnan(imputed_X_train).sum())
                        #print("Total de NaNs antes da restauração em imputed_X_test:", np.isnan(imputed_X_test).sum())

                        # Transforma em DataFrame para facilitar a manipulação
                        imputed_X_train = pd.DataFrame(imputed_X_train, columns=df_copy_X_train.columns, index=df_copy_X_train.index)
                        imputed_X_test = pd.DataFrame(imputed_X_test, columns=df_copy_X_test.columns, index=df_copy_X_test.index)

                        # Restaura os valores originais, mantendo apenas os imputados
                        imputed_X_train = df_copy_X_train.where(~mask_train, imputed_X_train)
                        imputed_X_test = df_copy_X_test.where(~mask_test, imputed_X_test)

                        # Transforma novamente em array para manter compatibilidade
                        imputed_X_train = imputed_X_train.to_numpy()
                        imputed_X_test = imputed_X_test.to_numpy()

                    
                    elif imputer_name == "LPMD2":
                        # Treinar e imputar com PMIVAE
                        lpmd2_model = MyPipeline.model_lpmd2(X_train, y_train)  # Treina o modelo
                        imputed_X_train = lpmd2_model.transform(X_train)  # Imputa os dados de treino
                        imputed_X_test = lpmd2_model.transform(X_test)  # Imputa os dados de teste

                    elif imputer_name == "LPMD":
                        # Treinar e imputar com PMIVAE
                        lpmd_model = MyPipeline.model_lpmd(X_train, y_train)  # Treina o modelo
                        imputed_X_train = lpmd_model.transform(X_train)  # Imputa os dados de treino
                        imputed_X_test = lpmd_model.transform(X_test)  # Imputa os dados de teste
                
                    else:
                        imputer.fit(X_train)
                        imputed_X_train = imputer.transform(X_train)
                        imputed_X_test = imputer.transform(X_test)


                prep.fit(imputed_X_train)
                
                best.fit(prep.transform(imputed_X_train), y_train)
                
                y_pred.extend(best.predict(prep.transform(imputed_X_test)))
                y_true.extend(y_test) 

                score[algorithm].append(recall_score(y_true, y_pred, labels=[0,1], average=None))
                aucscore = roc_auc_score(y_test, (best.predict_proba(prep.transform(imputed_X_test)))[:, 1])
                auc_score[algorithm].append(aucscore)

        
        elapsed_time = time.time() - start_time  # Calcula o tempo de execução
        
        # Armazena os tempos
        processing_times.append({
            "Dataset": name,
            "Imputer": imputer_name,
            "Tempo (s)": elapsed_time
        })

        
        # Salvar os resultados para cada imputer
        result_prefix = f"{name}_{imputer_name}"  

        auc_df = pd.DataFrame.from_dict(auc_score)
        auc_df.to_csv(result_prefix + '_auc.csv')

        recall_svm = pd.DataFrame(np.vstack(score['SVM']))
        recall_gb = pd.DataFrame(np.vstack(score['GB']))
        recall_rf = pd.DataFrame(np.vstack(score['RF']))

        esp = pd.concat([recall_svm[[0]], recall_rf[[0]], recall_gb[[0]]], axis=1)
        sen = pd.concat([recall_svm[[1]], recall_rf[[1]], recall_gb[[1]]], axis=1)

        esp.columns = ['SVM', 'RF', 'GB']
        sen.columns = ['SVM', 'RF', 'GB']

        esp.to_csv(result_prefix + '_spe.csv')
        sen.to_csv(result_prefix + '_sen.csv')  

# Criar e salvar a planilha com os tempos de processamento
df_times = pd.DataFrame(processing_times)
df_times.to_csv("tempos_processamento.csv", index=False)

fim = time.time()
print('\n')
print('Tempo de execução', round((fim - ini)/60, 4), 'minutos')


