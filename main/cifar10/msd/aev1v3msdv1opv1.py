import lightning as pl
import os
import sys
from optuna.integration import PyTorchLightningPruningCallback

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from models.cifar10 import AeV1V3MsdV1
from models.cifar10 import AeV1V3
from datamodules import CIFAR10Dm
from utils import transfer_weights
from utils import init_envir
from utils import load_pre_ae_model
from utils import optuna_main


def main(trial,
         bash_log_name,
         normal_class,
         seed,
         pre_epochs,
         epochs,
         log_path,
         objective,
         radio,
         batch_size,
         devices=2,
         enable_progress_bar=False):
    monitor = "svdd_roc_auc_sk"
    mse_loss_weight = trial.suggest_categorical("mse_loss_weight", [0.2])
    lnr_svdd = AeV1V3MsdV1(seed=seed,
                           mse_loss_weight=mse_loss_weight,
                           objective=objective,
                           visual=False)
    transfer_weights(lnr_svdd, auto_enc)
    # lnr_svdd.eval()
    lnr_svdd.init_center_c(lnr_svdd, cifar10.train_dataloader())
    # print(lnr_svdd.center)
    trainer = pl.Trainer(
        accelerator="gpu",
        #  strategy='ddp',
        devices=1,
        deterministic=True,
        # check_val_every_n_epoch=epochs,
        default_root_dir=log_path,
        max_epochs=epochs,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor)],
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)
    trainer.fit(model=lnr_svdd, datamodule=cifar10)
    return trainer.callback_metrics[monitor].item()


if __name__ == '__main__':
    args = init_envir()
    cifar10 = CIFAR10Dm(batch_size=args.batch_size,
                        seed=args.seed,
                        radio=args.radio,
                        normal_class=args.normal_class)
    auto_enc = AeV1V3.load_from_checkpoint(
        load_pre_ae_model(bash_log_name=args.bash_log_name,
                          batch_size=args.batch_size,
                          radio=args.radio,
                          n_epochs=args.pre_epochs,
                          seed=args.seed,
                          normal_class=args.normal_class,
                          model_name="aev1v3"))
    optuna_main(args=args, main=main, file=__file__)
