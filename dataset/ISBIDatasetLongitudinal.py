import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import feature
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.dataset_utils import Phase, Modalities, Mode, retrieve_data_dir_paths, Evaluate


class ISBIDatasetLongitudinal(Dataset):
    """ISBIDatasetLongitudinal dataset"""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patients=None,
                 evaluate: Evaluate = Evaluate.TRAINING, size=300, n_classes=5, preprocess=True, view=None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.phase = phase
        self.size = size
        self.n_classes = n_classes
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patients,
                                                      Mode.LONGITUDINAL, size, view)
        self.transforms = transforms.Compose([transforms.RandomRotation(10),
                                              transforms.RandomAffine((0, 0), translate=(0,0.25))])  # use for augmentation

    def __len__(self):
        return len(self.data_dir_paths)

    def crop_center(self, img, cropx, cropy):
        z, y, x = img.shape
        startx = x // 2 - (cropx // 2)
        return img[:, :cropy, startx:startx + cropx]

    def __getitem__(self, idx):
        x_ref, x, ref_label, label = [], [], None, None
        x_ref_path, x_path = self.data_dir_paths[idx]

        slice = int(x_path.split("/")[-1])
        view = int(x_path.split("/")[-2])
        try:
            if idx + 1 >= self.__len__():  # is_last is used for LTPR, LFPR and VD metrics -- can be omitted it from the code if not using these metrics
                is_last = True
            else:
                next_one = self.data_dir_paths[idx + 1][1]
                next_slice = int(next_one.split("/")[-1])
                is_last = next_slice <= slice and view == 2
        except:
            is_last = True
            print("Exception in extracting next slice")
        for i, modality in enumerate(self.modalities):
            with h5py.File(os.path.join(x_ref_path, f'{modality.value}.h5'), 'r') as f:
                x_ref.append(f['data'][()])
                if ref_label is None:
                    ref_label = torch.as_tensor(f['label'][()], dtype=torch.int64)
                    ref_label[ref_label > self.n_classes - 1] = self.n_classes - 1
                    ref_label = F.one_hot(ref_label, num_classes=self.n_classes).permute(2, 0, 1)
            with h5py.File(os.path.join(x_path, f'{modality.value}.h5'), 'r') as f:
                x.append(f['data'][()])
                if label is None:
                    try:
                        label = torch.as_tensor(f['label'][()], dtype=torch.int64)
                        label[label > self.n_classes - 1] = self.n_classes - 1
                        label = F.one_hot(label, num_classes=self.n_classes).permute(2, 0, 1)  # volume
                    except Exception:
                        return self.__getitem__(idx + 1)
        mismatch = []
        is_mismatch = False  # For patients with 3 scans, scan 2 is always referenced by scan 1 (hence the mismatch), scan 3 by scan 2, and scan 1 by scan 2.
        mismatch_path = None
        if self.data_dir_paths[idx][0].__contains__("2_3"):
            mismatch_path = self.data_dir_paths[idx][0].replace("2/2_3", "1/1_3")
            for i, modality in enumerate(self.modalities):
                with h5py.File(os.path.join(mismatch_path, f'{modality.value}.h5'), 'r') as f:
                    mismatch.append(f['data'][()])
                    is_mismatch = True

        x = np.array(x)
        x_ref = np.array(x_ref)
        if x.shape != (1, self.size, self.size):
            print("INCORRECT SHAPE", x_path, x.shape, label.shape, flush=True)
            x = resize(x, (1, self.size, self.size))
            label = resize(label, (self.n_classes, self.size, self.size), order=0)
        if x_ref.shape != (1, self.size, self.size):
            print("INCORRECT SHAPE", x_ref_path, x_ref.shape, ref_label.shape, flush=True)
            x_ref = resize(x_ref, (1, self.size, self.size))
            ref_label = resize(ref_label, (self.n_classes, self.size, self.size), order=0)

        if not len(mismatch):
            mismatch = x
        else:
            mismatch = np.array(mismatch)
        if mismatch.shape != (1, self.size, self.size):
            print("INCORRECT SHAPE mismatch", mismatch_path, mismatch.shape, flush=True)
            mismatch = resize(mismatch, (1, self.size, self.size))

        mismatch = torch.as_tensor(mismatch)
        x = torch.as_tensor(x)
        x_ref = torch.as_tensor(x_ref)

        return x_ref.float(), x.float(), torch.as_tensor(ref_label).float(), torch.as_tensor(
            label).float(), mismatch.float(), is_mismatch, is_last
