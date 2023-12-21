import os
import sys
import time
import pytorch_lightning as pl

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from models.cifar10 import AeV1V3
from utils import init_envir
from datamodules.cifar10 import CIFAR10Dm


def cifar10_lenet(bash_log_name,
                  normal_class,
                  seed,
                  pre_epochs,
                  epochs,
                  log_path,
                  radio,
                  batch_size,
                  devices=1,
                  enable_progress_bar=False):
    # log_path = log_path + datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
    datamodule = CIFAR10Dm(
        batch_size=batch_size,
        seed=seed,
        radio=radio,
        normal_class=normal_class,
    )
    auto_enc = AeV1V3(seed=seed, visual=args.visual)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         deterministic=True,
                         check_val_every_n_epoch=epochs,
                         default_root_dir=log_path,
                         max_epochs=epochs,
                         enable_progress_bar=enable_progress_bar,
                         enable_model_summary=False)
    trainer.fit(model=auto_enc, datamodule=datamodule)


if __name__ == '__main__':

    start_time = time.perf_counter()

    args = init_envir()
    cifar10_lenet(bash_log_name=args.bash_log_name,
                  normal_class=args.normal_class,
                  pre_epochs=args.pre_epochs,
                  epochs=args.epochs,
                  seed=args.seed,
                  radio=args.radio,
                  batch_size=args.batch_size,
                  enable_progress_bar=args.progress_bar,
                  log_path=args.log_path,
                  devices=args.devices)

    end_time = time.perf_counter()

    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
