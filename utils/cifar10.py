import argparse
import os
import torch
import numpy as np
import time
import optuna
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


def optuna_main(args, main, file=__file__):
    start_time = time.perf_counter()
    samplers = {
        'tpe': optuna.samplers.TPESampler(seed=args.seed),
        'random': optuna.samplers.RandomSampler(seed=args.seed)
    }
    pruners = {
        'median': optuna.pruners.MedianPruner(),
        'hyper': optuna.pruners.HyperbandPruner()
    }
    sampler = samplers[args.sampler]
    pruner = pruners[args.pruner]
    if args.progress_bar:
        if os.path.exists(args.opt_path):
            os.remove(args.opt_path)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///%s" % args.opt_path,
        engine_kwargs={"connect_args": {
            "timeout": 1000
        }})
    study = optuna.create_study(direction="maximize",
                                sampler=sampler,
                                pruner=pruner,
                                storage=storage,
                                load_if_exists=True,
                                study_name="class_%s" % args.normal_class)
    study.optimize(lambda trial: main(trial,
                                      bash_log_name=args.bash_log_name,
                                      normal_class=args.normal_class,
                                      pre_epochs=args.pre_epochs,
                                      epochs=args.epochs,
                                      seed=args.seed,
                                      radio=args.radio,
                                      batch_size=args.batch_size,
                                      enable_progress_bar=args.progress_bar,
                                      log_path=args.log_path,
                                      objective=args.objective,
                                      devices=args.devices),
                   n_trials=args.n_trials)
    value_msg = (f"Best value: {study.best_value} "
                 f"(params: {study.best_params})")
    sampler_pruner = (f"sampler: {study.sampler.__class__.__name__}, "
                      f"pruner: {study.pruner.__class__.__name__}")
    print(value_msg + "\n" + sampler_pruner)
    # end_time = time.process_time()
    end_time = time.perf_counter()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
