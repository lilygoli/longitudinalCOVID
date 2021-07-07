import json
import pprint
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd


def write_config(content, fname):
    with fname.open('wt') as handle:
        handle.write("CONFIG = " + pprint.pformat(content))
        handle.close()


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None, logger=None):
        self.writer = writer
        self.logger = logger
        keys = ["loss", "precision0", "precision1", "precision2", "precision3", "precision4",
                "recall0", "recall1", "recall2", "recall3", "recall4",
                "dice_loss", "dice_score0", "dice_score1", "dice_score2", "dice_score3", "dice_score4",
                "asymmetric_loss"
                ]
        keys += ["precision_difference0", "precision_difference1", "precision_difference2",
                 "recall_difference0", "recall_difference1", "recall_difference2",
                 "dice_loss_difference", "dice_score_difference0", "dice_score_difference1", "dice_score_difference2",
                 "asymmetric_loss_difference"
                 ]
        keys += ["precision_difference_reverse0", "precision_difference_reverse1", "precision_difference_reverse2",
                 "recall_difference_reverse0", "recall_difference_reverse1", "recall_difference_reverse2",
                 "dice_loss_difference_reverse", "dice_score_difference_reverse0", "dice_score_difference_reverse1",
                 "dice_score_difference_reverse2",
                 "asymmetric_loss_difference_reverse"
                 ]
        self.keys = keys

        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, is_last=False):
        if self.writer is not None:
            try:
                for i, v in enumerate(value):
                    if v is None:
                        continue
                    self.writer.add_scalar(key + str(i), v)
                    self._data.total[key + str(i)] += v * n
                    self._data.counts[key + str(i)] += n
                    self._data.average[key + str(i)] = self._data.total[key + str(i)] / self._data.counts[key + str(i)]

            except Exception as e:

                if value is None:
                    return
                self.writer.add_scalar(key, value)

                self._data.total[key] += value * n
                self._data.counts[key] += n
                self._data.average[key] = self._data.total[key] / self._data.counts[key]

        # if is_last:  #use for obtaining separate results for each patient
        #     self.logger.info("End of Volume!")
        #     for key in self.keys:
        #         self.logger.info('    {:15s}: {}'.format(str(key), self._data.average[key]))
        #     self.reset()

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



