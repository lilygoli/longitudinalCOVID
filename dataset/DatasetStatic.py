import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
from skimage.transform import resize
from torch.utils.data import Dataset
from pathlib import Path
from skimage import feature
from torchvision.transforms import transforms

from dataset.dataset_utils import Phase, Modalities, Views, Mode, retrieve_data_dir_paths, Evaluate


class DatasetStatic(Dataset):
    """DatasetStatic dataset"""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patients=None, evaluate: Evaluate = Evaluate.TRAINING, preprocess=True, size=300, n_classes=5,
                 view: Views = None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.size = size
        self.n_classes = n_classes
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patients, Mode.STATIC, size, view)

    def __len__(self):
        return len(self.data_dir_paths)

    def crop_center(self, img, cropx, cropy):
        z, y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:, starty:starty + cropy, startx:startx + cropx]

    def __getitem__(self, idx):

        data, label = [], None
        slice = int(self.data_dir_paths[idx].split("/")[-1])
        view = int(self.data_dir_paths[idx].split("/")[-2])
        try:
            if idx + 1 >= self.__len__():
                is_last = True
            else:
                next_one = self.data_dir_paths[idx + 1]
                next_slice = int(next_one.split("/")[-1])
                is_last = next_slice <= slice and view == 2

        except Exception as e:
            print("IS_LAST Exception", e)
            is_last = True
        for i, modality in enumerate(self.modalities):
            try:
                with h5py.File(os.path.join(self.data_dir_paths[idx], f'{modality.value}.h5'), 'r') as f:
                    data.append(f['data'][()])
                    if label is None:
                        label = f['label'][()]
                        label[label > self.n_classes - 1] = self.n_classes - 1
                        label = F.one_hot(torch.as_tensor(label, dtype=torch.int64), num_classes=self.n_classes).permute(2, 0, 1)
            except Exception as e:
                print("EXCEPTION in loading data!: ", e)
                return self.__getitem__(idx+1)
        mismatch, mismatch_label = [], None
        print(self.data_dir_paths[idx], flush=True)
        is_one = False
        if self.data_dir_paths[idx].__contains__("2_2"):
            mismatch_path = self.data_dir_paths[idx].replace("2/2_2", "1/1_2")
            is_one = True
        elif self.data_dir_paths[idx].__contains__("1_1"):
            mismatch_path = self.data_dir_paths[idx].replace("1/1_1", "2/2_1")
        else:
            mismatch_path = self.data_dir_paths[idx].replace("3/3_3", "2/2_3")

        for i, modality in enumerate(self.modalities):
            with h5py.File(os.path.join(mismatch_path, f'{modality.value}.h5'), 'r') as f:
                mismatch.append(f['data'][()])
                mismatch_label = torch.as_tensor(f['label'][()], dtype=torch.int64)
                mismatch_label[mismatch_label > self.n_classes - 1] = self.n_classes - 1
                mismatch_label = F.one_hot(mismatch_label, num_classes=self.n_classes).permute(2, 0, 1)

        data = np.array(data)
        if data.shape != (1,self.size, self.size):
            print("INCORRECT SHAPE", self.data_dir_paths[idx], data.shape, label.shape, flush=True)
            data = resize(data,(1,self.size, self.size))
            label = resize(label, (self.n_classes, self.size, self.size), order=0)

        mismatch = np.array(mismatch)
        if mismatch.shape != (1,self.size, self.size):
            print("INCORRECT SHAPE mismatch", mismatch_path, mismatch.shape, mismatch_label.shape , flush=True)
            mismatch = resize(mismatch, (1,self.size, self.size))
            mismatch_label = resize(mismatch_label, (self.n_classes, self.size, self.size), order=0)
        mismatch = torch.as_tensor(mismatch)
        data = torch.as_tensor(data).float()

        return data.float(), torch.as_tensor(label).float(), mismatch.float(), torch.as_tensor(mismatch_label).float(), is_one, is_last
