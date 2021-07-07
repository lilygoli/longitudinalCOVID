from abc import abstractmethod

import numpy as np
import torch

from base import BaseTrainer
from logger import Mode
from utils import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader, fold=None,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config, fold)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader,
        self.do_validation = self.valid_data_loader is not None
        self.do_training = self.data_loader is not None

        if self.valid_data_loader.__class__.__name__ == 'tuple':
            self.valid_data_loader = self.valid_data_loader[0]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader) if self.do_training else 0
            self.len_epoch_val = len(self.valid_data_loader) if self.do_validation else 0

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) if self.do_training else int(np.sqrt(valid_data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer, logger= self.logger)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer, logger=self.logger)

    @abstractmethod
    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        raise NotImplementedError('Method _process() from Trainer class has to be implemented!')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.do_training:

            self.model.train()
            self.train_metrics.reset()

            self._process(epoch, self.data_loader, self.train_metrics, Mode.TRAIN)

        log = self.train_metrics.result()

        if self.do_validation:

            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.do_training and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            self._process(epoch, self.valid_data_loader, self.valid_metrics, Mode.VAL)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def log_scalars(self, metrics, step, output, target, loss, mode=Mode.TRAIN, reverse=False, difference= False, toy=False,
                    is_last=None):
        if is_last is None:
            is_last = [False] * target.size(0)
        if not difference:
            self.writer.set_step(step, mode)
            if loss is not None:
                 metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                if met.__name__ in ["vd, LTPR, LFPR"]:
                    for i in range(target.size(0)):
                        if not is_last[i]:
                            met(output[i].unsqueeze(0), target[i].unsqueeze(0), is_last[i])
                            continue
                        metrics.update(met.__name__, met(output[i].unsqueeze(0), target[i].unsqueeze(0)))
                metrics.update(met.__name__, met(output, target))
        elif not reverse:
            self.writer.set_step(step, mode)
            for met in self.metric_ftns:
                if met.__name__ in ["LFPR", "LTPR"]:
                    continue
                if met.__name__ in ["vd"]:
                    for i in range(target.size(0)):
                        if not is_last[i]:
                            met(output[i].unsqueeze(0), target[i].unsqueeze[0], is_last[i])
                            continue
                        metrics.update(met.__name__, met(output[i].unsqueeze(0), target[i].unsqueeze(0)))
                metrics.update(met.__name__ + "_difference", met(output, target))
        else:
            self.writer.set_step(step, mode)
            last_metric = self.metric_ftns[-1].__name__
            for met in self.metric_ftns:
                if met.__name__ in ["LFPR","LTPR"]:
                    continue
                if met.__name__ in ["vd"]:
                    for i in range(target.size(0)):
                        if not is_last[i]:
                            met(output[i].unsqueeze(0), target[i].unsqueeze(0), is_last)
                            continue
                        metrics.update(met.__name__, met(output[i].unsqueeze(0), target[i].unsqueeze(0)))
                if met.__name__  in [last_metric]:
                    metrics.update(met.__name__ + "_difference_reverse", met(output, target), is_last=is_last)
                metrics.update(met.__name__ + "_difference_reverse", met(output, target))


    @staticmethod
    def _progress(data_loader, batch_idx, batches):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = batches
        return base.format(current, total, 100.0 * current / total)

    @staticmethod
    def get_step(batch_idx, epoch, len_epoch):
        return (epoch - 1) * len_epoch + batch_idx
