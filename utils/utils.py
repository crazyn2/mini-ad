import torch
import numpy as np
import random
import os
import sys
import yaml
import pandas as pd
import glob

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True


def transfer_weights(dst_net, src_net):
    """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

    dst_net_dict = dst_net.state_dict()
    src_net_dict = src_net.state_dict()

    # Filter out decoder network keys
    src_net_dict = {k: v for k, v in src_net_dict.items() if k in dst_net_dict}
    # Overwrite values in the existing state_dict
    dst_net_dict.update(src_net_dict)
    # Load the new state_dict
    dst_net.load_state_dict(dst_net_dict)


def load_pre_ae_model(bash_log_name: str,
                      batch_size: int,
                      radio: float,
                      n_epochs: int,
                      seed: int,
                      normal_class: int,
                      dataset="cifar10",
                      objective="ae",
                      model_name="ae") -> str:
    """ 
    Args:
    hyperparamters of model

    Returns:
    model path
    """
    path = "%s/%s/%s/%s/batch_size%d/radio%.2f/n_epochs%d/**/lightning_logs/version_0/hparams.yaml" % (
        bash_log_name, dataset, objective, model_name, batch_size, radio,
        n_epochs)
    hparams_files = glob.glob(path, recursive=True)
    # print(hparams_files)
    # print(path)
    keys = ['model_path', 'normal_class', 'seed']
    hypers_table = pd.DataFrame(columns=keys)

    for hparams_file in hparams_files:
        tmp_dict = {"model_path": os.path.dirname(hparams_file)}
        with open(hparams_file, 'r') as file_handle:
            hparams = yaml.full_load(file_handle)
            tmp_dict.update({k: v for k, v in hparams.items() if k in keys})
            hypers_table = pd.concat(
                [hypers_table, pd.DataFrame(tmp_dict, index=[0])],
                ignore_index=True)
    line = hypers_table[(hypers_table['seed'] == seed)
                        &
                        (hypers_table['normal_class'] == normal_class)].iloc[0]

    model_path = glob.glob(os.path.join(line['model_path'], "**/*.ckpt"))
    return model_path[0]


if __name__ == '__main__':
    print(
        load_pre_ae_model("bash-logv3",
                          100,
                          0,
                          200,
                          5,
                          7,
                          objective="ae",
                          model_name="aev1v9"))
