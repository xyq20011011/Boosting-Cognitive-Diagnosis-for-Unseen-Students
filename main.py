import os
from trainer import Trainer
from Models.AE_CDs import AE_NCD, Emb_NCD, VAE_NCD
from utils import *



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    Here is an example for training the AE-NeuralCD. 
    If you need to train a traditional model such as Emb_NCD, please use trainer.train_traditional().
    """
    model = VAE_NCD
    dataset_name = "Junyi"
    exp_name = f"{get_model_name(model)}_{dataset_name}"
    print(exp_name)

    trainer = Trainer(exp_name)  # Initialize Trainer
    trainer.verbose = False  # If trainer.verbose = False, then the detailed information for each epoch will not be displayed.
    trainer.load_data(name=dataset_name)
    trainer.init_model(model)

    epoch_num = 30
    trainer.train(to_epoch=epoch_num)







