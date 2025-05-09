from typing import Any, Dict, Tuple

import torch
import pytorch_lightning as pl
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from ..utils.tabCNN_metrics import pitch_precision, pitch_recall, pitch_f_measure, tab_precision, tab_recall, tab_f_measure, tab_disamb
from ..models.components.tabCNN import TabCNN

class TabCNNModule(pl.LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        spec_size=128
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if net is not None:
            self.net = net
        else:
            self.net = TabCNN(spec_size=spec_size)
        
        self.num_strings = 6

        # loss function
        self.criterion = self.catcross_by_string

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for averaging metrics across validation batches
        self.val_pp = MeanMetric()
        self.val_pr = MeanMetric()
        self.val_pf = MeanMetric()
        self.val_tp = MeanMetric()
        self.val_tr = MeanMetric()
        self.val_tf = MeanMetric()
        self.val_disamb = MeanMetric()

        # for averaging metrics across test batches
        self.test_pp = MeanMetric()
        self.test_pr = MeanMetric()
        self.test_pf = MeanMetric()
        self.test_tp = MeanMetric()
        self.test_tr = MeanMetric()
        self.test_tf = MeanMetric()
        self.test_disamb = MeanMetric()

        # for tracking best so far validation accuracy
        #self.val_acc_best = MaxMetric()

    def custom_categorical_cross_entropy(self, predictions, targets, epsilon=1e-9):
        N = predictions.shape[0]
        p = torch.log(predictions)
        mul = p * targets
        s = torch.sum(mul, dim=2)
        s = torch.sum(s, dim=0)
        s = torch.sum(s, dim=0)
        s = -1.0 * (s/N)
        #if torch.isnan(s).any():
        #    print("p stats:", p.min().item(), p.max().item(), p.mean().item(), p.var().item())
        #    print("targets stats:", targets.min().item(), targets.max().item(), targets.mean().item(), targets.var().item())

            
     

        return s

    def catcross_by_string(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        # Transform y from one hot to class encoding
        #is_no_class = y.sum(dim=2) == 0  # This will be True for rows that are all zeros
        #class_indices = y.argmax(dim=2)
        #class_indices[is_no_class] = y.shape[2]  # This assumes 0-based indexing, thus setting to 'j'
        
        #print('x: {}'.format(x.shape))
        #print('y: {}'.format(y.shape))
        
        return self.custom_categorical_cross_entropy(x, y)
        self.custom_categorical_cross_entropy(x, y)

        for i in range(self.num_strings):

            #loss += self.custom_categorical_cross_entropy(x, y)
            loss += F.cross_entropy(x[:, i, :], y[:, i])
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_pp.reset()
        self.val_pr.reset()
        self.val_pf.reset()
        self.val_tp.reset()
        self.val_tr.reset()
        self.val_tf.reset()
        self.val_disamb.reset()

        self.test_loss.reset()
        self.test_pp.reset()
        self.test_pr.reset()
        self.test_pf.reset()
        self.test_tp.reset()
        self.test_tr.reset()
        self.test_tf.reset()
        self.test_disamb.reset()
        

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        x, y = batch
        preds = self.forward(x)

        # Pad y so that it has same dimensions as preds
        padAmount = preds.shape[2] - y.shape[2]
        if padAmount > 0:
            y = F.pad(y, (0, padAmount))


        loss = self.criterion(preds, y)
        #if torch.isnan(loss).any():
        #    print(preds)
        #preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        print(targets)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_pp(pitch_precision(preds, targets))
        self.val_pr(pitch_recall(preds, targets))
        self.val_pf(pitch_f_measure(preds, targets))
        self.val_tp(tab_precision(preds, targets))
        self.val_tr(tab_recall(preds, targets))
        self.val_tf(tab_f_measure(preds, targets))
        self.val_disamb(tab_disamb(preds, targets))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pp", self.val_pp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pr", self.val_pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pf", self.val_pf, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tp", self.val_tp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tr", self.val_tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tf", self.val_tf, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tdr", self.val_disamb, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_pp(pitch_precision(preds, targets))
        self.test_pr(pitch_recall(preds, targets))
        self.test_pf(pitch_f_measure(preds, targets))
        self.test_tp(tab_precision(preds, targets))
        self.test_tr(tab_recall(preds, targets))
        self.test_tf(tab_f_measure(preds, targets))
        self.test_disamb(tab_disamb(preds, targets))
        self.log("test/pp", self.test_pp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pr", self.test_pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pf", self.test_pf, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tp", self.test_tp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tr", self.test_tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tf", self.test_tf, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tdr", self.test_disamb, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    #def on_train_batch_start(self, batch, batch_idx):
    #    #print("Before update:")
    #    for name, param in self.named_parameters():
    #        self.log('model_params/' + name, param.data.norm().item(), on_step=True, on_epoch=True, prog_bar=False)
    #        #print(f"{name}: {param.data.norm().item()}")


if __name__ == "__main__":
    _ = TabCNNModule(None, None, None, None)