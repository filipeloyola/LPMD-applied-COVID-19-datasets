#########################################################
# modules
#########################################################

import sklearn as sk
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.impute import SimpleImputer
#from sklearn.cluster import AgglomerativeClustering

# MICE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# K-means
from sklearn.cluster import KMeans

#########################################################
# functions
#########################################################

def compara_original(A, B):
    ''' Substitui os valores de A pelos valores de B, onde B for diferente de np.nan
        Args:
        A: Um np.array, referente a Y atual
        B: Um np.array com as mesmas dimensões de A, referente aos dados iniciais de referência
        Returns:
        Um np.array com os valores de A substituídos.
    '''
    A[~np.isnan(B)] = B[~np.isnan(B)]
    return A

# função do algoritmo principal
def label_propagation(data_missing, data_mice, epsilon, max_iter):
    ''' data_missing: np.array contendo os missings
        data_mice: np.array com imputação pelo método escolhido
        retorn: np.array preenchido por Label Propagation
        obs: essa função deve ser chamada por um cluster por vez
        epsilon: argumento do algoritmo
        max_iter: número máximo de iterações
    '''
    
    # definindo dataref
    dataref = data_missing.copy()

    # definindo matriz Y
    Y = data_mice.copy()

    # Fazendo matriz W
    W = rbf_kernel(Y) # Kernel RBF do sklearn com gamma=20
   
    # Fazendo matriz transição T 
    normalizer = W.sum(axis=0)     # Inicializa a variável normalizer com a soma de cada coluna
    W /= normalizer[:, np.newaxis] # Cada valor de W é dividido pela variável normalizar referente a cada coluna
    T = W

    Y_prev = np.zeros((Y.shape[0], Y.shape[1]))

    iteracoes = 0

    for n_iter in range(max_iter):

        iteracoes += 1 

        dif = np.abs(Y - Y_prev).sum()
        print(dif)

        if dif < epsilon:
            break

        Y_prev = Y

        Y = safe_sparse_dot(T,Y)

        Y = compara_original(Y, dataref) # voltando para o valor original, no caso de não ser missing
    
    print("Número de iterações: ", iteracoes)

    return(Y)


# função de filtro interquartis
def remover_outliers_iqr(df):
  """
  Remove outliers em cada coluna de um dataframe Pandas utilizando o IQR.

  Argumentos:
    df: Dataframe Pandas.

  Retorna:
    Dataframe Pandas com outliers substituidos por np.nan.
  """

  for coluna in df.columns:
    # Cálculo do primeiro e terceiro quartil
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)

    # Cálculo do IQR
    iqr = q3 - q1

    # Definição dos limites inferior e superior
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    # Substituição de outliers por np.nan
    df.loc[df[coluna] < limite_inferior, coluna] = np.nan
    df.loc[df[coluna] > limite_superior, coluna] = np.nan

  return df


#######################################################################################################################################
# Função para verificar colunas que tenha 100% de valores nulos. Coloca lixo no lugar, para a coluna não apagar durante o preenchimento
#######################################################################################################################################

def preencher_valores_nulos(dataframe):
  """
  Verifica todas as colunas de um DataFrame e preenche valores nulos com 0.

  Argumentos:
    dataframe: O DataFrame a ser analisado.

  Retorna:
    O DataFrame original com os valores nulos preenchidos por 0.
  """

  # Verifica se existem colunas com 100% de valores nulos
  for coluna in dataframe.columns:
    if dataframe[coluna].isnull().all():
      # Preenche os valores nulos com 0
      dataframe[coluna].fillna(0, inplace=True)

  return dataframe

# Exemplo de uso
dataframe_original = pd.DataFrame({
  'Coluna 1': [1, 2, np.nan],
  'Coluna 2': [4, 5, 6],
  'Coluna 3': [np.nan, np.nan, np.nan]
})



#########################################################
# Label propagation regression 8 - Era o 8 e agora é o 2 (LPR2) - LPMD2
#########################################################
#
# Label Propagation Regression
#
# Imputação inicial por média da classe (FIT)
#    
# Realiza filtro por intervavo interquatil (substitue os outlier por missing)
#
# COM divisão de classes
#    
# Imputação secundária por Média (Transform)
#
# Realiza o filtro no PREDICT
#
#########################################################

class LPMD2():
    ''' Recebe dataframe de dados com o missing
        Retorna dados completos
    '''
    # -------------------------------------------------------------------------------
    def __init__(self,  iteracoes=500, epsilon=1e-32):
    # Parâmetros
        self.max_iter = iteracoes
        print("max interações: ", self.max_iter )
        self.epsilon = epsilon
        print("epsilon: ", epsilon)

        # entrada dos dados completos para fazer medida de erro
        #self.data = datacomplete

    # -------------------------------------------------------------------------------
    def fit(self, datamissing, train_target):       

        # entrada de dados com missing
        self.dataMiss = datamissing

        # entrada de target (somente treino)
        self.train_target = pd.DataFrame(train_target)
        print(type(self.train_target))
        print(self.train_target)
        # Número de clusters
        print("Teste UNIQUE")
        print(self.train_target.iloc[:, 0].unique())

        list_groups = self.train_target.iloc[:, 0].unique()
        self.numero_clusters = len(list_groups)

        # criando dataset de referencia
        self.dataref = self.dataMiss

        # fazendo o agrupamento utilizando o target
        agrupamento = self.train_target.iloc[:, 0]

        # adicionando coluna com informação de cluster no dataset original
        self.dataMiss = pd.DataFrame(self.dataMiss)
        self.dataMiss["grupo"] = agrupamento
        Y_ok = self.dataMiss.copy(deep=True)

        # criando dataset que irá receber os valores imputados
        self.dataImp = self.dataMiss.copy(deep=True)
        print("dataImp: ", self.dataImp.shape)

        # criando objeto de imputador pela média
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        # lista auxiliar para receber grupos
        groups = self.dataImp["grupo"].unique()

        # processo de imputação para cada classe
        for i in groups:

            # somente a porção de determinada classe
            df_imputed_classe = self.dataImp.loc[self.dataImp['grupo'] == i]
            cols = df_imputed_classe.columns
            print("df_imputed_classe: ", type(df_imputed_classe))
            print(df_imputed_classe.shape)
            print(df_imputed_classe)

            # passa pelo filtro de interquartis
            df_imputed_classe = remover_outliers_iqr(df_imputed_classe.copy())

            # verifica se alguma coluna possui 100% de missing. Se sim, insere lixo no lugar. Foi necessário, para que essa coluna não suma
            df_imputed_classe = preencher_valores_nulos(df_imputed_classe.copy())

            # fit imputer
            mean = imputer.fit(df_imputed_classe)

            # transform
            dataImp_classe = mean.transform(df_imputed_classe)
            print("Tipos de dado: ", type(dataImp_classe))
            print(dataImp_classe)

            # recebe os valores imputados
            self.dataImp.loc[self.dataImp['grupo'] == i] = dataImp_classe


        # Realizando o Label Propagation Regression para cada classe
        for i in groups:
           
            # somente a porção de determinada classe
            df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
            df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

            # deleta informação de cluster
            df_imputed = df_imputed_cluster.drop(columns=["grupo"])
            df_missing = df_missing_cluster.drop(columns=["grupo"])

            # transforma em np.array
            df_imputed = df_imputed.to_numpy()
            df_missing = df_missing.to_numpy()

            # chama Label Propagation
            retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
            
            # transforma retorno (np.array) em pandas dataframe
            retorno_lp = pd.DataFrame(retorno_lp)

            # colunas do dataframe
            colunas = self.dataMiss.columns
            colunas = colunas[:-1] # colunas sem o cluster

            # dataset para receber dataset de entrada somente com determinado cluster
            df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

            # passando o indice de um para outro
            retorno_lp.index = df_entrada.index
            
            # dataset de entrada recebe saída do label propagation
            df_entrada[colunas] = retorno_lp

            # dataset de saída recebe informações atualizadas
            Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
        
        # retira informação de target
        Y_ok = Y_ok.drop(columns="grupo")

        # Dataframe preenchido
        self.treino_preenchido = Y_ok

        return(self)

    # -------------------------------------------------------------------------------    
    def transform(self, datamissing, is_Train=False):

        if (is_Train):
            Y_ok = self.treino_preenchido

            # de dataframe para numpy array
            Y_ok = Y_ok.to_numpy()

        else: 
            # entrada de dados com missing
            self.dataMiss = datamissing

            # verifica o tamanho dos dados de treino
            tamanho_dados_treino = len(self.treino_preenchido)

            # junta os dados de treino (imputados) com os dados de teste (sem imputar)
            X_sem_imputar = np.concatenate([self.treino_preenchido, self.dataMiss])
            
            # criando dataset de referencia
            self.dataref =  X_sem_imputar

            # imputando através de MEAN
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            mean = imputer.fit(X_sem_imputar)
            self.dataImp = mean.transform(X_sem_imputar)
            
            # fazendo o agrupamento por fast greedy
            kmeans = KMeans(n_clusters = self.numero_clusters)
            kmeans.fit(self.dataImp)
            agrupamento = kmeans.labels_

            # adicionando coluna com informação de cluster no dataset original
            self.dataMiss = pd.DataFrame(X_sem_imputar)
            self.dataMiss["grupo"] = agrupamento
            Y_ok = self.dataMiss.copy(deep=True)

            # adicionando coluna com informação de cluster no dataset dataImp
            self.dataImp = pd.DataFrame(self.dataImp)
            self.dataImp["grupo"] = agrupamento

            # lista auxiliar para receber grupos
            groups = self.dataImp["grupo"].unique()

            # Realizando o Label Propagation Regression para cada cluster
            for i in groups:
            
                # somente a porção de determinada classe
                df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
                df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

                # deleta informação de cluster
                df_imputed = df_imputed_cluster.drop(columns=["grupo"])
                df_missing = df_missing_cluster.drop(columns=["grupo"])

                # passa pelo filtro de interquartis
                df_imputed = remover_outliers_iqr(df_imputed.copy())

                # preenche os valores filtrados com a média 
                mean = imputer.fit(df_imputed)

                # transform (imputer)
                df_imputed = mean.transform(df_imputed.copy())

                # transforma em np.array
                #df_imputed = df_imputed.to_numpy() # quando faz o imputed já retorna numpy
                df_missing = df_missing.to_numpy()

                # chama Label Propagation
                retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
                
                # transforma retorno (np.array) em pandas dataframe
                retorno_lp = pd.DataFrame(retorno_lp)

                # colunas do dataframe
                colunas = self.dataMiss.columns
                colunas = colunas[:-1] # colunas sem o cluster

                # dataset para receber dataset de entrada somente com determinado cluster
                df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

                # passando o indice de um para outro
                retorno_lp.index = df_entrada.index
                
                # dataset de entrada recebe saída do label propagation
                df_entrada[colunas] = retorno_lp

                # dataset de saída recebe informações atualizadas
                Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
            
            # retira informação de target
            Y_ok = Y_ok.drop(columns="grupo")

            # de dataframe para numpy array
            Y_ok = Y_ok.to_numpy()

            # Fatia o dataset imputado concatenado para obter apenas os dados de teste (output_md_test)
            Y_ok = Y_ok[tamanho_dados_treino:]


        return(Y_ok)




####################################################################################################
# Label propagation regression 3 (LPR3) - é uma cópia do 0, porém como inicialização igual a -1.
####################################################################################################
#
# Label Propagation Regression
#
# Imputação inicial por -1
#
# COM divisão de classes no Label Propagation
#    
# Imputação secundária por média total (transform)
#
#########################################################

class LabelPropagationRegression3():
    ''' Recebe dataframe de dados com o missing
        Retorna dados completos
    '''
    # -------------------------------------------------------------------------------
    def __init__(self, datacomplete, iteracoes=500, epsilon=1e-32):
    # Parâmetros
        self.max_iter = iteracoes
        print("max interações: ", self.max_iter )
        self.epsilon = epsilon
        print("epsilon: ", epsilon)

        # entrada dos dados completos para fazer medida de erro
        self.data = datacomplete

    # -------------------------------------------------------------------------------
    def fit(self, datamissing, train_target):       

        # entrada de dados com missing
        self.dataMiss = datamissing

        # entrada de target (somente treino)
        self.train_target = pd.DataFrame(train_target)

        # Número de clusters
        list_groups = self.train_target[0].unique()
        self.numero_clusters = len(list_groups)

        # criando dataset de referencia
        self.dataref = self.dataMiss

        # fazendo o agrupamento utilizando o target
        agrupamento = self.train_target[0]

        # adicionando coluna com informação de cluster no dataset original
        self.dataMiss = pd.DataFrame(self.dataMiss)
        self.dataMiss["grupo"] = agrupamento
        Y_ok = self.dataMiss.copy(deep=True)

        # criando dataset que irá receber os valores imputados
        self.dataImp = self.dataMiss.copy(deep=True)

        # realiza a imputação de -1 no lugar de valores ausentes
        self.dataImp.fillna(value=-1, inplace=True)
        
        # lista auxiliar para receber grupos
        groups = self.dataImp["grupo"].unique()

        print("Imputado:")
        print(self.dataImp.isnull().sum())


        # Realizando o Label Propagation Regression para cada classe
        for i in groups:
           
            # somente a porção de determinada classe
            df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
            df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

            # deleta informação de cluster
            df_imputed = df_imputed_cluster.drop(columns=["grupo"])
            df_missing = df_missing_cluster.drop(columns=["grupo"])

            # transforma em np.array
            df_imputed = df_imputed.to_numpy()
            df_missing = df_missing.to_numpy()

            # chama Label Propagation
            retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
            
            # transforma retorno (np.array) em pandas dataframe
            retorno_lp = pd.DataFrame(retorno_lp)

            # colunas do dataframe
            colunas = self.dataMiss.columns
            colunas = colunas[:-1] # colunas sem o cluster

            # dataset para receber dataset de entrada somente com determinado cluster
            df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

            # passando o indice de um para outro
            retorno_lp.index = df_entrada.index
            
            # dataset de entrada recebe saída do label propagation
            df_entrada[colunas] = retorno_lp

            # dataset de saída recebe informações atualizadas
            Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
        
        # retira informação de target
        Y_ok = Y_ok.drop(columns="grupo")

        # Dataframe preenchido
        self.treino_preenchido = Y_ok

        return(self)

    # -------------------------------------------------------------------------------    
    def transform(self, datamissing, is_Train=False):

        if (is_Train):
            Y_ok = self.treino_preenchido

            # de dataframe para numpy array
            Y_ok = Y_ok.to_numpy()

        else: 
            # entrada de dados com missing
            self.dataMiss = datamissing

            # verifica o tamanho dos dados de treino
            tamanho_dados_treino = len(self.treino_preenchido)

            # junta os dados de treino (imputados) com os dados de teste (sem imputar)
            X_sem_imputar = np.concatenate([self.treino_preenchido, self.dataMiss])
            
            # criando dataset de referencia
            self.dataref =  X_sem_imputar

            # imputando através de média
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            mean = imputer.fit(X_sem_imputar)
            self.dataImp = mean.transform(X_sem_imputar)
            
            # fazendo o agrupamento por fast greedy
            kmeans = KMeans(n_clusters = self.numero_clusters)
            kmeans.fit(self.dataImp)
            agrupamento = kmeans.labels_

            # adicionando coluna com informação de cluster no dataset original
            self.dataMiss = pd.DataFrame(X_sem_imputar)
            self.dataMiss["grupo"] = agrupamento
            Y_ok = self.dataMiss.copy(deep=True)

            # adicionando coluna com informação de cluster no dataset dataImp
            self.dataImp = pd.DataFrame(self.dataImp)
            self.dataImp["grupo"] = agrupamento

            # lista auxiliar para receber grupos
            groups = self.dataImp["grupo"].unique()

            # Realizando o Label Propagation Regression para cada cluster
            for i in groups:
            
                # somente a porção de determinada classe
                df_imputed_cluster = self.dataImp.loc[self.dataImp['grupo'] == i]
                df_missing_cluster = self.dataMiss.loc[self.dataMiss['grupo'] == i]

                # deleta informação de cluster
                df_imputed = df_imputed_cluster.drop(columns=["grupo"])
                df_missing = df_missing_cluster.drop(columns=["grupo"])

                # transforma em np.array
                df_imputed = df_imputed.to_numpy()
                df_missing = df_missing.to_numpy()

                # chama Label Propagation
                retorno_lp = label_propagation(df_missing, df_imputed, self.epsilon, self.max_iter)
                
                # transforma retorno (np.array) em pandas dataframe
                retorno_lp = pd.DataFrame(retorno_lp)

                # colunas do dataframe
                colunas = self.dataMiss.columns
                colunas = colunas[:-1] # colunas sem o cluster

                # dataset para receber dataset de entrada somente com determinado cluster
                df_entrada = Y_ok.loc[self.dataMiss['grupo'] == i]

                # passando o indice de um para outro
                retorno_lp.index = df_entrada.index
                
                # dataset de entrada recebe saída do label propagation
                df_entrada[colunas] = retorno_lp

                # dataset de saída recebe informações atualizadas
                Y_ok.loc[self.dataMiss['grupo'] == i] = df_entrada
            
            # retira informação de target
            Y_ok = Y_ok.drop(columns="grupo")

            # de dataframe para numpy array
            Y_ok = Y_ok.to_numpy()

            # Fatia o dataset imputado concatenado para obter apenas os dados de teste (output_md_test)
            Y_ok = Y_ok[tamanho_dados_treino:]


        return(Y_ok)



