
import numpy

from logger import Mode
from trainer.ISBITrainer import ISBITrainer
from utils.illustration_util import log_visualizations
import torch.nn.functional as F
import torch


class LongitudinalTrainer(ISBITrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader, fold=None,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config, data_loader, fold, valid_data_loader, lr_scheduler,
                         len_epoch)

    def remap_labels_for_difference(self, output):
        covid_noncovid_output_ref = output.argmax(1)
        covid_noncovid_output_ref2 = covid_noncovid_output_ref.clone()
        covid_noncovid_output_ref2[covid_noncovid_output_ref != 3] = 0
        covid_noncovid_output_ref2[covid_noncovid_output_ref == 3] = 1
        covid_noncovid_output_ref[covid_noncovid_output_ref >= 2] = 3
        covid_noncovid_output_ref[covid_noncovid_output_ref <= 1] = 0
        covid_noncovid_output_ref[covid_noncovid_output_ref == 3] = 1

        # first mask is for covid/non-covid difference  and second mask is for cons/non-cons
        return covid_noncovid_output_ref, covid_noncovid_output_ref2

    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        _len_epoch = self.len_epoch if mode == Mode.TRAIN else self.len_epoch_val
        TOY = False  # set to True to perform a toy experiment where target and reference CTs are the same

        for batch_idx, (x_ref, x, target_ref, target, mismatch, is_mismatch, is_last) in enumerate(data_loader):
            x_ref, x, target, target_ref = x_ref.to(self.device), x.to(self.device), target.to(
                self.device), target_ref.to(self.device)

            if mode == Mode.TRAIN:
                self.optimizer.zero_grad()
            if not TOY:
                output, encoded = self.model(x_ref, x)
            else:
                output, encoded = self.model(x,x)
            loss = self.loss(output, target)

            if mode == Mode.TRAIN:
                loss.backward()
                self.optimizer.step()

            self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), output, target, loss, mode,
                             is_last=is_last, toy=TOY)

            covid_noncovid_output, covid_noncovid_output2 = self.remap_labels_for_difference(output)
            covid_noncovid_target, covid_noncovid_target2 = self.remap_labels_for_difference(target)
            covid_noncovid_target_ref, covid_noncovid_target_ref2 = self.remap_labels_for_difference(target_ref)

            difference_output = None
            difference_output_reverse = None
            difference_target = None
            difference_target_reverse = None
            output_ref = None

            if mode == Mode.VAL:
                mismatch = mismatch.to(self.device)
                if not TOY:
                    output_ref, encoded_ref = self.model(mismatch, x_ref)
                else:
                    output_ref, encoded_ref = self.model(x_ref, x_ref)

                covid_noncovid_output_ref, covid_noncovid_output_ref2 = self.remap_labels_for_difference(output_ref)

                difference_output = covid_noncovid_output - covid_noncovid_output_ref
                difference_output += 1

                difference_output_reverse = covid_noncovid_output2 - covid_noncovid_output_ref2
                difference_output_reverse += 1

                difference_target = covid_noncovid_target - covid_noncovid_target_ref
                difference_target += 1

                difference_target_reverse = covid_noncovid_target2 - covid_noncovid_target_ref2
                difference_target_reverse += 1

                d_output = F.one_hot(difference_output, num_classes=3).permute(0, 3, 1, 2)
                d_target = F.one_hot(difference_target, num_classes=3).permute(0, 3, 1, 2)
                d_target_reverse = F.one_hot(difference_target_reverse, num_classes=3).permute(0, 3, 1, 2)
                d_output_reverse = F.one_hot(difference_output_reverse, num_classes=3).permute(0, 3, 1, 2)
                try:
                    output_refs = torch.tensor([]).to(self.device)
                    target_refs = torch.tensor([]).to(self.device)
                    empty = True
                    for i in range(x.size(0)):
                        if not is_mismatch[i]:
                            empty = False
                            output_refs = torch.cat((output_refs, output_ref[i].unsqueeze(0)))
                            target_refs = torch.cat((target_refs, target_ref[i].unsqueeze(0)))

                    if not empty:
                        self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), output_refs, target_refs,
                                     None, mode, False, is_last=is_last, toy=TOY)

                except Exception as e:
                    print("Exception in mismatch:", is_mismatch, e)

                self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), d_output, d_target, None,
                                 mode, False, True, is_last=is_last, toy=TOY)
                self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), d_output_reverse,
                                 d_target_reverse, None,
                                 mode, True, True, is_last=is_last, toy=TOY)

            if not (batch_idx % self.log_step):
                self.logger.info(f'{mode.value} Epoch: {epoch} {self._progress(data_loader, batch_idx,_len_epoch)} Loss: {loss.item():.6f}')
            if not (batch_idx % (_len_epoch // 10)):
                log_visualizations(self.writer, x_ref, x, output, target, output_ref, target_ref,
                                   difference_output, difference_target, difference_output_reverse,
                                   difference_target_reverse, encoded, toy=TOY)

            del x_ref, x, target, target_ref, mismatch



