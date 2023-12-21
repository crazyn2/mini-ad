from torch import nn
from torchmetrics import AUROC
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models.base.ae import BaseAe
from models.base.ae import Ae
# from addict import Dict


class Aev1V1Encoder(nn.Module):

    def __init__(self, rep_dim):
        super().__init__()
        self.rep_dim = rep_dim

        # Encoder (must match the Deep SVDD network above)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Aev1V1Decoder(nn.Module):

    def __init__(self, rep_dim):
        '''Construct is same as cifar10_LeNet'''
        super().__init__()
        # Decoder
        self.rep_dim = rep_dim
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        # self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)),
        #                                   128,
        #                                   5,
        #                                   bias=False,
        #                                   padding=2)
        # self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        # self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        # self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
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

        x = self.bn1d(x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class AeV1V1(BaseAe):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=1e-4,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        # super().__init__(seed, rep_dim, lr, weight_decay, lr_milestone,
        #  optimizer_name, visual)
        super().__init__(
            seed=seed,
            rep_dim=rep_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )
        self.encoder = Aev1V1Encoder(self.rep_dim)
        self.decoder = Aev1V1Decoder(self.rep_dim)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('leaky_relu'))
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        return Ae(dec_out=dec_x, enc_out=enc_x)


class AeV1V2(AeV1V1):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=1e-4,
        weight_decay=0.5e-6,
        # rep_dim,
        # lr,
        # weight_decay,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        # super().__init__(seed, rep_dim, lr, weight_decay, lr_milestone,
        #  optimizer_name, visual)
        super().__init__(
            seed=seed,
            rep_dim=rep_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs +
                       torch.randn(inputs.size(), device=inputs.device))
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        mse_loss = self.mse(inputs, dec_out)
        if self.visual:
            mse_loss_scores = torch.sum((dec_out - inputs)**2,
                                        dim=tuple(range(1, dec_out.dim())))
            self.training_step_outputs.append(mse_loss_scores)
        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)
        self.log("train_loss", mse_loss)
        # self.log_tsne(outputs["dec_out"], self.current_epoch)
        return {'loss': mse_loss}


class AeV1V3(BaseAe):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=1e-4,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        # super().__init__(seed, rep_dim, lr, weight_decay, lr_milestone,
        #  optimizer_name, visual)
        super().__init__(
            seed=seed,
            rep_dim=rep_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )

        # Encoder (must match the Deep SVDD network above)
        self.pool = nn.MaxPool2d(2, 2)
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

        return Ae(dec_out=dec_x, enc_out=enc_x)


