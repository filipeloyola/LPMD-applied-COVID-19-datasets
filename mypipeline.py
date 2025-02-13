
# Partial Multiple Imputation with Variational Autoencoders
from Autoencoders.pmivae import PMIVAE
from Autoencoders.vae_pmivae import ConfigVAE

# Label Propagation Regression
from LPMD import *


# Siamese Autoencoder
from Autoencoders.saei import ConfigSAE, SAEImp, DataSets


# Ignorar todos os avisos
import warnings
warnings.filterwarnings("ignore")


# ==========================================================================
class MyPipeline:

    def model_autoencoder_pmivae(dataset_train):
        original_shape = dataset_train.shape

        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_fc = [0.1]
        vae_config.latent_dimension = 5
        vae_config.input_shape = (original_shape[1],)

        pmivae_model = PMIVAE(vae_config, num_samples=200)
        model = pmivae_model.fit(dataset_train)

        return model

    # ------------------------------------------------------------------------
    def modelo_saei(
        dataset_train,
        dataset_test,
        dataset_train_md,
        dataset_test_md,
        nome_coluna,
        input_shape,
    ):
        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        pmivae_model = SAEImp()

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=dataset_train_md.fillna(np.mean(dataset_train[nome_coluna])),
            x_val_pre=dataset_test_md.fillna(np.mean(dataset_test[nome_coluna])),
        )

        model = pmivae_model.fit(dados, vae_config)
        return model
    
     # ------------------------------------------------------------------------
    def model_lpmd2(
        datamissing,
        target
    ):
        imputer = LPMD2(iteracoes=1000, epsilon=1e-256)
        model = imputer.fit(datamissing, target)
        return model