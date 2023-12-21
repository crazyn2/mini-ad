from __future__ import print_function
import os
import sys
import torch
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
# import pytorch_lightning as pl
import lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
# from keras.preprocessing.image import apply_affine_transform
from random import sample
import random
import itertools
import time
from torchvision.datasets import CIFAR10
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from preprocessing import global_contrast_normalization
from utils import get_target_label_idx


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR10Dm(pl.LightningDataModule):
    __doc__ = r"""Initialize cifar10 cfg.

        .. seealso::
            See :attr:`~dataset.fmnist` for related property.

        Args:
            batch_size:
                batch_size parameter of dataloader
            normal_class:
                normal class which's labelled 0
            seed:
                dataloader workers's initial seed
            radio:
                rate of abnormal samples that were classified as normalities

    """

    def __init__(
        self,
        batch_size,
        normal_class,
        seed,
        radio=0.0,
        num_workers=3,
        root="./data/",
        dataset_name="cifar10",
        gcn=True,
    ):
        """Initialize cifar10 cfg.

        .. seealso::
            See :attr:`~dataset.fmnist` for related property.

        Args:
            batch_size:
                batch_size parameter of dataloader
            normal_class:
                normal class which's labelled 0
            seed:
                dataloader workers's initial seed
            radio:
                rate of abnormal samples that were classified as normalities

        """
        super().__init__()
        # normal class only one class per training set
        self.save_hyperparameters()
        pl.seed_everything(seed, workers=True)
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.root = root
        # 污染数据比例
        self.radio = radio
        self.normal_class = normal_class
        self.num_workers = num_workers
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.seed = seed
        self.gcn = gcn
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        # def prepare_data(self):
        #     # download
        #     CIFAR10(self.root, train=True, download=True)
        #     CIFAR10(self.root, train=False, download=True)

        # def setup(self, stage: str) -> None:

        # Pre-computed min and max values (after applying GCN)
        # from train data per class
        # global_contrast_normalization
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        gcn_transform = [
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ]
        transforms_list = [transforms.ToTensor()]
        if self.gcn:
            transforms_list += gcn_transform

        else:
            transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))

        transform = transforms.Compose(transforms_list)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(
        #         lambda x: global_contrast_normalization(x, scale='l1')),
        #     transforms.Normalize([min_max[self.normal_class][0]] * 3, [
        #         min_max[self.normal_class][1] - min_max[self.normal_class][0]
        #     ] * 3)
        # ])
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        # if stage == "fit":
        train_cifar10 = CIFAR10(
            root=self.root,
            train=True,
            transform=transform,
            # download=True,
            target_transform=target_transform,
        )

        train_indices = [
            idx for idx, target in enumerate(train_cifar10.targets)
            if target in self.normal_classes
        ]
        dirty_indices = [
            idx for idx, target in enumerate(train_cifar10.targets)
            if target not in self.normal_classes
        ]
        train_indices += sample(
            dirty_indices,
            int(len(train_indices) * self.radio / (1 - self.radio)))
        # dataloader shuffle=True will mix the order of normal and abnormal
        # extract the normal class of cifar10 train dataset
        self.train_cifar10 = Subset(train_cifar10, train_indices)

        # if stage == "test":
        self.test_cifar10 = CIFAR10(
            root=self.root,
            train=False,
            transform=transform,
            # download=True,
            target_transform=target_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker,
                          generator=self.gen,
                          persistent_workers=True,
                          shuffle=True,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_cifar10,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=self.gen,
            #   shuffle=True,
            drop_last=True)

    def load_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=self.gen,
            #   shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            self.test_cifar10,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=self.gen,
            #   shuffle=True,
            drop_last=True)


def my_affine_tch(img, flip, tx, ty, k_90_rotate):
    # img = img.copy()
    if flip:
        img = torch.flip(img, dims=[1])
    if k_90_rotate != 0:
        img = F.affine(img,
                       scale=1,
                       shear=0,
                       angle=k_90_rotate,
                       translate=(tx, ty),
                       fill=0)

    return img


class CIFAR10GmTranV1(CIFAR10Dm):

    def __init__(
        self,
        batch_size,
        normal_class,
        seed,
        radio=0,
        num_workers=4,
        root="./data/",
        dataset_name="cifar10",
        gcn=False,
        max_tx=8,
        max_ty=8,
    ):
        super().__init__(
            batch_size=batch_size,
            normal_class=normal_class,
            seed=seed,
            radio=radio,
            num_workers=num_workers,
            root=root,
            dataset_name=dataset_name,
            gcn=gcn,
        )
        self.max_tx = max_tx
        self.max_ty = max_ty

    def setup(self, stage: str) -> None:
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.947014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        gcn_transform = [
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ]
        transforms_list = [transforms.ToTensor()]
        if self.gcn:
            transforms_list += gcn_transform

        else:
            transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
        # transforms.RandomAffine
        transform = transforms.Compose(transforms_list)
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))
        self.test_cifar10 = CIFAR10(root=self.root,
                                    train=False,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)

        if stage == "fit":
            self.train_datasets = []
            self.test_datasets = []
            gmtran_class = 0
            for flip, tx, ty, k_90_rotate in itertools.product(
                (False, True), (0, -self.max_tx, self.max_tx),
                (0, -self.max_ty, self.max_ty), range(4)):

                subtrain_cifar10 = CIFAR10(
                    root=self.root,
                    train=True,
                    transform=transform,
                    # target_transform=target_transform,
                    download=False)

                subtrain_cifar10.data = np.load(
                    "./data/gmtran/data_cifar10_%d.npy" % gmtran_class)
                subtrain_cifar10.targets = np.load(
                    "./data/gmtran/targets_cifar10_%d.npy" % gmtran_class)
                self.train_datasets.append(subtrain_cifar10)

                subtrain_test_cifar10 = CIFAR10(
                    root=self.root,
                    train=False,
                    transform=transform,
                    target_transform=target_transform,
                    download=False)

                subtrain_test_cifar10.data = np.load(
                    "./data/gmtran/test_data_cifar10_%d.npy" % gmtran_class)
                subtrain_test_cifar10.targets = np.load(
                    "./data/gmtran/test_targets_cifar10_%d.npy" % gmtran_class)
                self.test_datasets.append(subtrain_test_cifar10)
                gmtran_class += 1
                # break
            self.train_cifar10 = ConcatDataset(self.train_datasets)
            self.test_cifar10 = ConcatDataset(self.test_datasets)


class CIFAR10GmTranV2(CIFAR10Dm):

    def __init__(
        self,
        batch_size,
        normal_class,
        seed,
        radio=0,
        num_workers=4,
        root="./data/",
        dataset_name="cifar10",
        gcn=False,
        max_tx=8,
        max_ty=8,
    ):
        super().__init__(
            batch_size=batch_size,
            normal_class=normal_class,
            seed=seed,
            radio=radio,
            num_workers=num_workers,
            root=root,
            dataset_name=dataset_name,
            gcn=gcn,
        )
        self.max_tx = max_tx
        self.max_ty = max_ty

    def setup(self, stage: str) -> None:
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.947014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        gcn_transform = [
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ]
        transforms_list = [transforms.ToTensor()]
        if self.gcn:
            transforms_list += gcn_transform

        else:
            transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
        # transforms.RandomAffine
        transform = transforms.Compose(transforms_list)
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))
        self.test_cifar10 = CIFAR10(root=self.root,
                                    train=False,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)

        if stage == "fit":
            self.train_datasets = []
            self.test_datasets = []
            gmtran_class = 0
            for _, _, _, _ in itertools.product((False, True),
                                                (0, -self.max_tx, self.max_tx),
                                                (0, -self.max_ty, self.max_ty),
                                                range(4)):

                subtrain_cifar10 = CIFAR10(root=self.root,
                                           train=True,
                                           transform=transform,
                                           target_transform=target_transform,
                                           download=False)

                subtrain_cifar10.data = np.load(
                    "./data/gmtran/data_cifar10_%d.npy" % gmtran_class)
                subtrain_cifar10.targets = np.load(
                    "./data/gmtran/targets_cifar10_%d.npy" % gmtran_class)
                self.train_datasets.append(subtrain_cifar10)

                subtrain_test_cifar10 = CIFAR10(
                    root=self.root,
                    train=False,
                    transform=transform,
                    target_transform=target_transform,
                    download=False)

                subtrain_test_cifar10.data = np.load(
                    "./data/gmtran/test_data_cifar10_%d.npy" % gmtran_class)
                subtrain_test_cifar10.targets = np.load(
                    "./data/gmtran/test_targets_cifar10_%d.npy" % gmtran_class)
                self.test_datasets.append(subtrain_test_cifar10)
                gmtran_class += 1
                # break
            self.train_cifar10 = ConcatDataset(self.train_datasets)
            self.test_cifar10 = ConcatDataset(self.test_datasets)


class CIFAR10GmTranV3(CIFAR10Dm):

    def __init__(
        self,
        batch_size,
        normal_class,
        seed,
        radio=0,
        num_workers=4,
        root="./data/",
        dataset_name="cifar10",
        gcn=False,
        max_tx=8,
        max_ty=8,
        preload=True,
    ):
        super().__init__(
            batch_size=batch_size,
            normal_class=normal_class,
            seed=seed,
            radio=radio,
            num_workers=num_workers,
            root=root,
            dataset_name=dataset_name,
            gcn=gcn,
        )
        self.max_tx = max_tx
        self.max_ty = max_ty
        self.preload = preload

    def setup(self, stage: str) -> None:
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        gcn_transform = [
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]] * 3, [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ] * 3)
        ]
        affine_transforms = []

        if stage == "fit":
            self.train_datasets = []
            self.test_datasets = []
            # gmtran_class = 0
            for flip, tx, ty, k_90_rotate in itertools.product(
                (False, True), (0, -self.max_tx, self.max_tx),
                (0, -self.max_ty, self.max_ty), range(4)):
                affine_transforms.append(
                    transforms.Lambda(lambda img: my_affine_tch(
                        img, flip, tx, ty, k_90_rotate)))
            for i in range(len(affine_transforms)):
                # geo_target_transform = transforms.Lambda(
                #     lambda x: 0 if x not in self.outlier_classes else i + 1)

                geo_target_transform = transforms.Lambda(lambda x: i)
                transform = [
                    transforms.Pad([16, 16], padding_mode='reflect'),
                    transforms.ToTensor()
                ] + [affine_transforms[i]]
                transform += [transforms.CenterCrop((32, 32))]
                # GCN
                if self.gcn:
                    transform += gcn_transform
                else:
                    transform += [transforms.Normalize([0.5] * 3, [0.5] * 3)]
                transform = transforms.Compose(transform)
                subtrain_cifar10 = CIFAR10(
                    root=self.root,
                    train=True,
                    transform=transform,
                    target_transform=geo_target_transform,
                    download=False)

                train_indices = [
                    idx for idx, target in enumerate(subtrain_cifar10.targets)
                    if target in self.normal_classes
                ]
                dirty_indices = [
                    idx for idx, target in enumerate(subtrain_cifar10.targets)
                    if target not in self.normal_classes
                ]
                train_indices += sample(
                    dirty_indices,
                    int(len(train_indices) * self.radio / (1 - self.radio)))
                # subtrain_cifar10.targets = [gmtran_class] * len(
                #     subtrain_cifar10.targets)
                subtrain_cifar10 = Subset(subtrain_cifar10, train_indices)
                self.train_datasets.append(subtrain_cifar10)
                # test
                # subtrain_test_cifar10 = CIFAR10(
                #     root=self.root,
                #     train=False,
                #     transform=transform,
                #     target_transform=target_transform,
                #     download=False)
                # for i in range(len(subtrain_test_cifar10.data)):

                #     subtrain_test_cifar10.data[i] = my_affine_tch(
                #         subtrain_test_cifar10.data[i], flip, tx, ty,
                #         k_90_rotate)
                # np.save(
                #     "./data/gmtran_tch/test_data_cifar10_%d.npy" %
                #     gmtran_class, subtrain_test_cifar10.data)
                # np.save(
                #     "./data/gmtran_tch/test_targets_cifar10_%d.npy" %
                #     gmtran_class, subtrain_test_cifar10.targets)
                # self.test_datasets.append(subtrain_test_cifar10)
                # gmtran_class += 1
            self.train_cifar10 = ConcatDataset(self.train_datasets)
            # self.test_cifar10 = ConcatDataset(self.test_datasets)

            # train_indices = [
            #     idx for idx, target in enumerate(train_cifar10.targets)
            #     if target in self.normal_classes
            # ]
            # dirty_indices = [
            #     idx for idx, target in enumerate(train_cifar10.targets)
            #     if target not in self.normal_classes
            # ]
            # train_indices += sample(
            #     dirty_indices,
            #     int(len(train_indices) * self.radio / (1 - self.radio)))
            # # dataloader shuffle=True will mix the order of normal and abnormal
            # # extract the normal class of cifar10 train dataset
            # self.train_cifar10 = Subset(train_cifar10, train_indices)
            # self.train_cifar10 = train_cifar10


def get_gcn():
    os.makedirs("bash-log/cifar10", exist_ok=True)

    # print(os.path.dirname(__file__))
    # print(os.path.dirname(os.path.dirname(__file__)))
    # i = 0
    # for inputs, labels in data_loader_train:
    #     print(inputs.shape)
    #     # plot_images_grid(inputs, export_img="log/cifar10/train_%d" % i)
    #     break
    #     i += 1
    train_set_full = CIFAR10(
        root="./data/",
        train=True,
        #  download=True,
        transform=None,
        target_transform=None)

    MIN = []
    MAX = []
    for normal_classes in range(10):
        train_idx_normal = get_target_label_idx(train_set_full.targets,
                                                normal_classes)
        train_set = Subset(train_set_full, train_idx_normal)

        _min_ = []
        _max_ = []
        for idx in train_set.indices:
            print(train_set.dataset.data[idx])
            gcm = global_contrast_normalization(
                torch.from_numpy(train_set.dataset.data[idx]).double(), 'l1')
            print(gcm)
            _min_.append(gcm.min())
            _max_.append(gcm.max())
            break
        MIN.append(np.min(_min_))
        MAX.append(np.max(_max_))
        break
    print(list(zip(MIN, MAX)))


def geo_tran():
    transform = transforms.Compose([
        transforms.Pad([105, 105], padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.CenterCrop((32, 32)),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    train_set = CIFAR10(
        root="./data/",
        train=True,
        #  download=True,
        transform=transform,
        target_transform=None)
    train_loader = DataLoader(train_set,
                              batch_size=64,
                              shuffle=True,
                              drop_last=True)
    for inputs, labels in train_loader:
        plot_images_grid(inputs, "bash-log/cifar10v1")
        # print(labels)
        break


if __name__ == '__main__':
    start_time = time.perf_counter()
    # geo_tran()
    # get_gcn()
    # cifar10 = CIFAR10Dm(batch_size=64,
    #                     normal_class=1,
    #                     radio=0.00,
    #                     gcn=False,
    #                     num_workers=1,
    #                     seed=0)
    # cifar10 = CIFAR10GmTranV1(batch_size=100,
    #                           normal_class=1,
    #                           radio=0.00,
    #                           seed=0,
    #                           gcn=False)
    cifar10 = CIFAR10GmTranV3(batch_size=64,
                              normal_class=1,
                              radio=0.00,
                              gcn=False,
                              seed=0)
    cifar10.setup("fit")
    train_data = cifar10.train_dataloader()
    # print(len(train_data))
    for inputs, labels in train_data:
        # print(inputs.shape)
        # plot_images_grid(inputs, "bash-log/cifar10v3", padding=0)
        print(labels)
        # break

    # cifar10 = CIFAR10GmTranV2(batch_size=100,
    #                           normal_class=1,
    #                           seed=0,
    #                           gcn=False)
    # cifar10.setup("fit")
    # train_data = cifar10.train_dataloader()
    # print(len(train_data))
    # for inputs, labels in train_data:
    #     plot_images_grid(inputs, "bash-log/cifar10v1")
    #     # print(labels)
    #     break
    # for inputs, labels in cifar10.test_dataloader():
    #     # plot_images_grid(inputs, "imgs/cifar10batch_test01")
    #     print(labels)
    #     break
    end_time = time.perf_counter()
    # end_time = time.process_time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
