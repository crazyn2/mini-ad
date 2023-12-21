import argparse
import os
import torch
import numpy as np
import random
from lightning.fabric import seed_everything
import lightning as pl


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def worker_init_fn(worker_id):
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def init_envir():
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="DSVDD")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opt_path', type=str, default="bash-log/db.sqlite3")
    parser.add_argument('--normal_class', type=int, default=0)
    parser.add_argument('--pre_epochs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--radio', type=float, default=0.0)
    parser.add_argument('--objective', type=str, default="one-class")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--log_path", type=str, default=os.getcwd())
    parser.add_argument("--bash_log_name", type=str, default="bash-logv3")
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--rep_dim", type=int, default=200)
    parser.add_argument("--sampler", type=str, default='random')
    parser.add_argument("--pruner", type=str, default='median')
    parser.add_argument("--preloaded", action="store_true")
    parser.add_argument("--monitor", type=str, default="mse")
    # parser.add_argument("--mariadb_name", type=str, default="ae")

    args = parser.parse_args()
    # seed_everything(args.seed, workers=False)
    seed_everything(args.seed, workers=True)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['PL_GLOBAL_SEED'] = str(args.seed)
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    return args
