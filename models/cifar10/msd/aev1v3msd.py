import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.msd import BaseMsdV1
from addict import Dict
from utils import get_radius
from models.base.ae import Ae


class AeV1V3MsdV1(BaseMsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """vanilla SVDD autoencoder
        """
        super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
                         weight_decay, lr_milestones, optimizer_name, visual,
                         objective)
        # 考虑下修改神经网络初始化方式
        # https://zhuanlan.zhihu.com/p/405752148
        # LSTM输入输出大小是否会影响结果
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.bn2d = nn.BatchNorm1d(self.rep_dim * 4 * 4,
                                   eps=1e-04,
                                   affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)),
                                          128,
                                          5,
                                          bias=False,
                                          padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        enc_x = self.fc1(x)
        x = self.bn1d(enc_x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        dec_x = torch.sigmoid(x)

        # return Dict({"dec_out": dec_x, 'enc_out': enc_x})
        return Ae(dec_out=dec_x, enc_out=enc_x)


class AeV1V3MsdV2(AeV1V3MsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        "aemsd"
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def init_center_c(self, net, train_loader, eps=0.1):
        # n_samples = 0
        # c = torch.zeros(self.rep_dim).cuda()

        centers = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs = net(inputs)
                dec_out = outputs.dec_out
                # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
                centers.append(dec_out)
                # n_samples += enc_out.shape[0]
                # c += torch.sum(enc_out, dim=0)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()
        # c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = svdd_loss
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        dec_out = outputs.dec_out
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        # l1 score
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        l1_svdd_scores = dec_out.sub(self.center).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        svdd_mse_scores = svdd_scores + self.mse_loss_weight * mse_scores
        # Save triples of (idx, label, score) in a list
        zip_params = [
            labels,
            svdd_scores,
            mse_scores,
            svdd_mse_scores,
            l1_mse_scores,
            l1_svdd_scores,
        ]
        if self.visual:
            # add additional record values
            zip_params += [dec_out, inputs]
        self.validation_step_outputs += list(zip(*zip_params))


class AeV1V3MsdV3(AeV1V3MsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ denoising + msd """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs +
                       torch.randn(inputs.size(), device=inputs.device))
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        mse_loss = self.mse(inputs, dec_out)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = svdd_loss + self.mse_loss_weight * mse_loss
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}


class AeV1V3MsdV4(AeV1V3MsdV3):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ denoising + msd + noising center"""
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def init_center_c(self, net, train_loader, eps=0.1):
        # n_samples = 0
        # c = torch.zeros(self.rep_dim).cuda()

        centers = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs = net(inputs +
                              torch.randn(inputs.size(), device=inputs.device))
                enc_out = outputs.enc_out
                # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
                centers.append(enc_out)
                # n_samples += enc_out.shape[0]
                # c += torch.sum(enc_out, dim=0)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()
        # c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c


class AeV1V3MsdV5(AeV1V3MsdV2):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ denoising + aemsd """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs +
                       torch.randn(inputs.size(), device=inputs.device))
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = svdd_loss
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}


class AeV1V3MsdV6(AeV1V3MsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ kl gaussian """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        if isinstance(outputs, dict):
            enc_out = outputs["enc_out"]
            dec_out = outputs["dec_out"]
        else:
            enc_out = outputs.enc_out
            dec_out = outputs.dec_out
            if self.global_step == 0:
                self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        gaussian_target = F.softmax(torch.randn(enc_out.size(),
                                                device=enc_out.device),
                                    dim=1)
        kl_loss = self.kl_divergence(enc_out, gaussian_target)
        mse_loss = self.mse(inputs, dec_out)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = (svdd_loss + self.mse_loss_weight * mse_loss +
                self.kl_loss_weight * kl_loss)
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}
