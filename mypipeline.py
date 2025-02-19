
# Partial Multiple Imputation with Variational Autoencoders
from Autoencoders.pmivae import ConfigVAE, PMIVAE
#from Autoencoders.pmivae import PMIVAE
#from Autoencoders.vae_pmivae import ConfigVAE


# Label Propagation Regression
from LPMD import *


# Siamese Autoencoder
from Autoencoders.saei import ConfigSAE, SAEImp, DataSets

# bibliotecas
import numpy as np
import pandas as pd


# Ignorar todos os avisos
import warnings
warnings.filterwarnings("ignore")


# ==========================================================================
class MyPipeline:

    def model_autoencoder_pmivae(dataset_train):
        original_shape = dataset_train.shape

        print("[PMIVAE] Training...")

        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 500
        vae_config.neurons = [10]
        vae_config.dropout_fc = [0.1]
        vae_config.latent_dimension = 3
        vae_config.input_shape = (original_shape[1],)
        vae_config.batch_size = 4

        pmivae_model = PMIVAE(vae_config, num_samples=100)
        model = pmivae_model.fit(dataset_train)

        return model


    # ------------------------------------------------------------------------
    def model_saei(
        dataset_train_md,
        dataset_test_md,
        col_name,
        input_shape,
    ):
        
        print("[SAEI] Training...")

        # initial dumb imputation
        dumb_imputer = SimpleImputer(strategy="mean")
        dumb_imputer.fit(dataset_train_md)

        dataset_train = dumb_imputer.transform(dataset_train_md)
        dataset_test = dumb_imputer.transform(dataset_test_md)

        # transforma em dataframe para operações posteriores
        dataset_train = pd.DataFrame(dataset_train, columns=dataset_train_md.columns)
        dataset_test = pd.DataFrame(dataset_test, columns=dataset_test_md.columns)

        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        saei_model = SAEImp()

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=dataset_train_md.fillna(np.mean(dataset_train[col_name])),
            x_val_pre=dataset_test_md.fillna(np.mean(dataset_test[col_name])),
        )

        model = saei_model.fit(dados, vae_config)
        return model
    
     # ------------------------------------------------------------------------
    def model_lpmd2(
        datamissing,
        target
    ):
        print("[LPMD2] Training...")

        imputer = LPMD2(iteracoes=1000, epsilon=1e-256)
        model = imputer.fit(datamissing, target)
        return model
    
    def model_lpmd(
        datamissing,
        target
    ):
        print("[LPMD] Training...")
        imputer = LPMD(iteracoes=1000, epsilon=1e-256)
        model = imputer.fit(datamissing, target)
        return model