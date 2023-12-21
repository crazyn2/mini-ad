from torch import nn
import lightning as pl
import io
import PIL.Image
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import namedtuple

Ae = namedtuple("Ae", ["enc_out", "dec_out"])
AeMem = namedtuple("AeMem", ["enc_out", "dec_out", "att"])
AeMemV2 = namedtuple("AeMem", ["enc_out", "mem_out", "dec_out", "att"])


class BaseAe(pl.LightningModule):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        # print(self.hparams)
        pl.seed_everything(seed, workers=True)
        self.rep_dim = rep_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestones
        self.optimizer_name = optimizer_name
        self.mse = nn.MSELoss(reduction='mean')
        self.visual = visual
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.aucroc_keys = ['mse', 'l1_mse']

    def training_step(self, train_batch, batch_idx):
        inputs, _ = train_batch
        outputs = self(inputs)
        if isinstance(outputs, dict):
            dec_out = outputs["dec_out"]
        else:
            dec_out = outputs.dec_out
            # if self.global_step == 0:
            #     self.logger.experiment.add_graph(self, inputs)
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

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        if isinstance(outputs, dict):
            dec_out = self(inputs)["dec_out"]
        else:
            dec_out = self(inputs).dec_out
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            inputs.size(0), -1).sum(dim=1, keepdim=False)

        zip_params = [labels, mse_scores, l1_mse_scores]
        if self.visual:
            # add additional record values
            zip_params += [inputs]
        self.validation_step_outputs += list(zip(*zip_params))

    def visual_hist(self, data: tuple, labels: tuple, title):
        fig, ax = plt.subplots()
        ax.set_title(title + '_hist')
        cat_data = np.concatenate(data)
        ax.hist(
            data,
            alpha=0.5,
            range=[cat_data.min(), np.percentile(cat_data, 99)],
            label=labels,
            # density=True,
            bins='auto')
        ax.legend()
        self.logger.experiment.add_figure(tag=title + '_hist',
                                          figure=fig,
                                          global_step=self.current_epoch)

    def on_validation_epoch_end(self):
        # torchmetrics
        # auroc = AUROC(task="binary")
        # auroc = AUROC(task="binary", average="none")
        unpacked_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpacked_labels_scores[0])
        labels_np = labels.cpu().data.numpy()
        for i in range(0, len(self.aucroc_keys)):
            scores = torch.stack(unpacked_labels_scores[i + 1])
            # torchmetrics aucroc
            # auroc_score_trh = auroc(scores, labels)
            # sklearn.metrics aucroc
            scores_np = scores.cpu().data.numpy()
            auroc_score_sk = roc_auc_score(labels_np, scores_np)
            # self.log(self.aucroc_keys[i] + '_roc_auc_trh',
            #          auroc_score_trh,
            #          prog_bar=True,
            #          sync_dist=True)
            self.log(self.aucroc_keys[i] + '_roc_auc_sk',
                     auroc_score_sk,
                     prog_bar=True,
                     sync_dist=True)
            if self.visual:
                self.logger.experiment.add_histogram(
                    tag=self.aucroc_keys[i] + '_scores',
                    values=torch.stack(unpacked_labels_scores[1]),
                    global_step=self.current_epoch)
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.visual_hist([scores_np_normal, scores_np_abnormal],
                                 ['0', '1'], self.aucroc_keys[i])
        if self.visual:
            # only when not empty
            if len(self.training_step_outputs) != 0:
                mse_scores_np = torch.stack(
                    unpacked_labels_scores[1]).cpu().data.numpy()
                training_scores_np = torch.cat(
                    self.training_step_outputs).cpu().data.numpy()
                self.logger.experiment.add_histogram(
                    tag='training_scores',
                    values=training_scores_np,
                    global_step=self.current_epoch)
                self.visual_hist(
                    [training_scores_np, mse_scores_np[labels_np == 0]],
                    ['train', 'test'], 'train_test_scores')
                self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

    def log_img(self, X_emb, labels, name, step):
        """Create a pyplot plot and save to buffer."""
        buf = io.BytesIO()
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=[labels])
        plt.savefig(buf, format='jpeg')
        plt.clf()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        self.logger.experiment.add_image(name, image, step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestone, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
