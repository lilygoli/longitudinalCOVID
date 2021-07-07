import os

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm

from dataset.rigid_and_deformable_registration import deformable_registration
from .util import (
    load_config_yaml,
    split_idxs,
    rm_tree,
    verify_config_hash,
    save_config_hash,
    crop_to_mask
)
import atexit
from sklearn.model_selection import StratifiedKFold, KFold
import nibabel as nib
import time
import sys


class DatasetPreprocessor:
    def __init__(
            self,
            config_yml_path, data_dir, size,
            force_preprocessing=True,
            cleanup=False,
            verbose=True,
            seed=42,
    ):
        # load dataset config

        config_yml_path = Path(config_yml_path)
        self.data_dir = data_dir
        self.size = size
        assert config_yml_path.is_file(), f"config yaml could not be found at '{config_yml_path}', {os.curdir}"
        self.cfg = load_config_yaml(config_yml_path)
        self.verbose = verbose

        if verbose:
            print(f"Loaded {self.cfg['name']} setup from {config_yml_path}", flush=True)

        # set data root dir
        # self.data_root = Path(data_root)
        self.pp_path = config_yml_path.parent
        assert (
            self.pp_path.is_dir()
        ), f"preprocessed directory could not be found at '{self.pp_path}'"

        # load setup config
        setup_yml_path = self.pp_path / "setup.yml"
        assert Path(
            setup_yml_path
        ).is_file(), f"setup yaml could not be found at '{config_yml_path}'"
        self.setup = load_config_yaml(setup_yml_path)
        if verbose:
            print(f"Loaded {self.setup['name']} setup from {setup_yml_path}", flush=True)

        # set temporary dir for npy files and csv
        self.npy_path = self.pp_path / ("npy_" + self.cfg["name"])
        self.tmp_df_path = self.npy_path / "tmp_df.csv"

        # setup cleanup
        if cleanup:
            atexit.register(self.cleanup)

        # load base patient dataframe
        self.df_path = self.pp_path / "base_df.csv"
        self.df = pd.read_csv(self.df_path)
        if verbose:
            print(f"Dataframe loaded from {self.df_path}")

        if "drop_na_col" in self.cfg.keys():
            if self.cfg["drop_na_col"] is not None:
                df = self.df.dropna(subset=[self.cfg["drop_na_col"]])
                self.df = df.reset_index(drop=True)

        # check if patients are selected manually
        if "manual_split" in self.cfg.keys():
            print("Manual splits detected, stored in 'manual_split'")
            self.df["manual_split"] = None
            for split, pat_ids in self.cfg["manual_split"].items():
                for pat_id in pat_ids:
                    self.df.loc[
                        self.df[self.setup["id_col"]] == pat_id, "manual_split"
                    ] = split
            # select only volumes that have a split assigned
            self.df.dropna(subset=["manual_split"], inplace=True)

        # temporary file in npy folder to lock only execute pre-processing once
        lock_file = (self.npy_path / 'lock')
        if lock_file.is_file():
            print('found lock file - waiting until pre-processing is finished', end='')
            while lock_file.is_file():
                time.sleep(5)
                print("sleeping")

                print('.', end='')
            print(' continuing')
        # exit()
        print("done making lock file", flush=True)
        # check if temporary dir exists already

        if (
                self.npy_path.is_dir()
                and not force_preprocessing
                and self.tmp_df_path.is_file()
                and verify_config_hash(config_yml_path, self.npy_path)
        ):
            if verbose:
                print(
                    f"npy folder found at {self.npy_path}! (delete folder for new preprocessing or set force_preprocessing)"
                )
            print(f"{self.setup['name']} '{self.cfg['name']}' preprocessed data found")
            self.df = pd.read_csv(self.tmp_df_path)

        else:

            try:
                self.npy_path.mkdir(exist_ok=force_preprocessing)
            except FileExistsError:
                print(
                    f"npy folder found at {self.npy_path}! (delete folder for new preprocessing or set force_preprocessing)"
                )
            # create lockfile
            lock_file.touch()

            # preprocess all data with npz files and safe npy
            print(f"Preprocessing {self.setup['name']} '{self.cfg['name']}'..")

            self._preprocess_all()

            # select only volumes that have been preprocessed
            df = self.df.dropna(subset=["dim0"])
            num_vol = len(df)
            num_pat = len(self.df)
            if num_vol < num_pat:
                print(
                    f"WARNING: only {num_vol} out of {num_pat} have been preprocessed. Dropping rest of entries.."
                )
            self.df = df.reset_index(drop=True)
            if 'manual_split' in self.cfg.keys():
                print('manual split found in config - skipping automatic splitting')
            else:
                if num_pat < 10:
                    print('less than 10 patients. 50-50 split in train and val')
                    test_size = 0
                    val_size = 0.5
                    train_size = 0.5
                else:
                    # SIMPLE TRAIN, VAL, TEST SPLITTING
                    test_size = self.cfg["test_size"]
                    val_size = self.cfg["val_size"]
                    train_size = 1 - val_size - test_size
                print(
                    f"Creating split 'train_val_test_split': {test_size:.0%} test, {val_size:.0%} val and {train_size:.0%} train"
                )
                splits = ["train", "val", "test"]
                idxs = np.arange(len(self.df))
                idxs_split = split_idxs(
                    idxs, test_size=test_size, val_size=val_size, seed=seed, shuffle=True
                )
                self.df["train_val_test_split"] = None
                for split in splits:
                    self.df.loc[idxs_split[split], "train_val_test_split"] = split

                # 5-FOLD-SPLIt
                if len(self.df) > 5:
                    stratify = self.cfg['stratify']
                    idxs = np.arange(len(self.df))
                    n_splits = 5

                    if stratify:
                        strat_label = np.zeros_like(idxs)
                        for i, label in enumerate(self.cfg['labels']):
                            strat_label += 2 ** i * (self.df[f"num_{label}"] > 0).to_numpy(dtype=int)
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        for k, (train_idx, val_idx) in enumerate(skf.split(idxs, strat_label)):
                            split_col = f"split_{n_splits}fold_{k}"
                            self.df[split_col] = None
                            self.df.loc[train_idx, split_col] = "train"
                            self.df.loc[val_idx, split_col] = "val"
                        strat_print = ", stratified"
                    else:
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                        for k, (train_idx, val_idx) in enumerate(kf.split(idxs)):
                            split_col = f"split_{n_splits}fold_{k}"
                            self.df[split_col] = None
                            self.df.loc[train_idx, split_col] = "train"
                            self.df.loc[val_idx, split_col] = "val"
                        strat_print = ''

                    print(
                        f"Created k-fold cross validation split: 'split_{n_splits}fold_k' - {n_splits}-fold, shuffle, seed 42{strat_print} - splits: 'train', 'val'"
                    )

                else:
                    print('Omitting 5-fold-split due to limited number of volumes')

            # copy config and create hash
            new_cfg_path = self.npy_path / Path(config_yml_path).name
            new_cfg_path.write_text(Path(config_yml_path).read_text())
            save_config_hash(new_cfg_path, self.npy_path)

            # save temporary dataframe
            self.df.to_csv(self.tmp_df_path)

            # remove lockfile
            lock_file.unlink()

            if verbose:
                print(f"Temporary data has been extracted to {self.npy_path}")
                print("Successfully preprocessed data")

    def print_cfg(self):
        # print config
        print("\nConfiguration:")
        print(yaml.dump(self.cfg))

    def export_config(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / 'patients.csv')
        (path / f'config_{self.cfg["name"]}.yml').write_text(yaml.dump(self.cfg))

    def cleanup(self):
        # cleaning up tmp dir
        rm_tree(self.npy_path)

    def _preprocess_all(self):
        """
        loop through all patients in df with npz files
        map channels and labels
        preprocess and save npy files

        """
        print("pre preprocessing started")

        cfg = self.cfg["preprocessing"]
        df = self.df

        for i in range(3):
            df[f"dim{i}"] = None
        for label in self.cfg["labels"]:
            df[f"num_{label}"] = 0
        # drop rows where selected labelmap is not present

        df.drop(
            df[
                df[f'nii_{self.cfg["labelmap"]}'] == False
                ].index, inplace=True)
        for idx, patient in tqdm(df.iterrows(), total=df.shape[0]):

            pat_id = patient[self.setup["id_col"]]

            tt = []

            for i in os.listdir(self.data_dir + f"/{pat_id}/preprocessed/"):
                if i.endswith(".nii"):
                    tt += [i]

            sys.stdout.flush()
            times = range(len(tt))

            datas, ps, p2s = [], [], []
            for time in times:

                # load data
                print("i", time, times, flush=True)

                p = self.data_dir+f"/{pat_id}/preprocessed/{pat_id}_0{time + 1}_simple_pp.nii"  # Change directory if needed
                data = nib.load(p).get_fdata()

                # load seg
                labelmap = self.cfg["labelmap"] if "labelmap" in self.cfg.keys() else "default"
                p = self.data_dir+f"/{pat_id}/pathMasks/{pat_id}_0{time + 1}_pathMask.nii"  # Change directory if needed
                p2 = self.data_dir+f"/{pat_id}/masks/{pat_id}_0{time + 1}_mask.nii"  # Change directory if needed

                try:
                    seg = nib.load(p).get_fdata()
                    lung = nib.load(p2).get_fdata()
                except:
                    continue

                label_counts = self._get_label_counts(seg)
                for k, v in label_counts.items():
                    self.df.loc[idx, f"num_{k}"] = np.array(v, dtype=np.uint64)
                datas += [data]
                ps += [seg]
                p2s += [lung]
            for time_1 in times:
                for time_2 in times:
                    # perform preprocessing (only done once)

                    cropped_data, cropped_seg, cropped_lung, mask = crop_to_mask(datas[time_1], ps[time_1], p2s[time_1])
                    cropped_data_ref, cropped_seg_ref, cropped_lung_ref, mask_ref = crop_to_mask(datas[time_2],
                                                                                                 ps[time_2],
                                                                                                 p2s[time_2])

                    data = cropped_data
                    seg = cropped_seg
                    lung = cropped_lung
                    data_ref = cropped_data_ref
                    seg_ref = cropped_seg_ref
                    lung_ref = cropped_lung_ref

                    data = self._clip(data, low_thr=cfg["clip_low_thr"], up_thr=cfg["clip_up_thr"])
                    data_ref = self._clip(data_ref, low_thr=cfg["clip_low_thr"], up_thr=cfg["clip_up_thr"])
                    # normalize
                    1

                    data = self._normalize(data, np.ones_like(data, dtype=bool))
                    data_ref = self._normalize(data_ref, np.ones_like(data_ref, dtype=bool))
                    data = resize(data, (self.size, self.size, self.size))
                    seg = resize(seg, (self.size, self.size, self.size), order=0)
                    lung = resize(lung, (self.size, self.size, self.size), order=0)

                    data_ref = resize(data_ref, (self.size, self.size, self.size))
                    seg_ref = resize(seg_ref, (self.size, self.size, self.size), order=0)
                    lung_ref = resize(lung_ref, (self.size, self.size, self.size), order=0)

                    # if np.histogram(seg, [0, 1, 2, 3, 4, 5])[0][1] == 0: # but some data have non-int values so we fix that first
                    com_seg = seg.astype(np.uint32)
                    com_lung = np.clip(np.round(lung), 0 , 1)
                    seg[com_seg + com_lung == 1] = 1
                    if time_1 != time_2:  # register time_1 data to time_2 data
                        lung, data, seg = deformable_registration(lung_ref,
                                                                  lung,
                                                                  data_ref,
                                                                  data,
                                                                  seg_ref,
                                                                  seg)

                    # save number of layers to df
                    for i in range(3):
                        self.df.loc[idx, f"dim{i}"] = np.array(
                            data.shape[i], dtype=np.uint64
                        )
                    # save to disk as npy

                    parent_dir = {"simple_pp": "preprocessed", "pathMask": "pathMasks", "mask": "masks"}
                    save_dict = {}
                    save_dict["simple_pp"] = data
                    save_dict["pathMask"] = seg
                    save_dict["mask"] = lung
                    for key in save_dict.keys():
                        path = Path(self.data_dir+ f"/{pat_id}/{parent_dir[key]}/{pat_id}_{time_1 + 1}-{time_2 + 1}_{key}")  # change directory
                        np.save(path.with_suffix(".npy"), save_dict[key])
                    print("saved", pat_id, time_1, time_2, flush=True)

    def _get_label_counts(self, seg):
        counts = {}
        for c, label in enumerate(self.cfg["labels"]):
            counts[label] = (seg == c).sum()

        return counts

    def _remap_channels(self, data):
        """map selected modalities to input channels"""
        channels = self.cfg["channels"]
        new_data = []
        for c, modality in enumerate(channels):
            new_data.append(np.expand_dims(data[modality], axis=0))
        new_data = np.hstack(new_data)
        return new_data

    def _remap_labels(self, seg, labelmap):
        """"map selected labels to segmentation map values"""
        new_seg = np.zeros(seg.shape, dtype=seg.dtype)
        for new_label_value, label_name in enumerate(self.cfg["labels"]):
            label_value = self.setup["labels"][labelmap][label_name]
            new_seg[seg == label_value] = new_label_value
            if self.cfg["labelmap"] == "quicknat" and label_name == "Cortical Grey Matter Right":
                new_seg[
                    (seg > 100) & (seg % 2 == 0)
                    ] = new_label_value
            if self.cfg["labelmap"] and label_name == "Cortical Grey Matter Left":
                new_seg[
                    (seg > 100) & (seg % 2 == 1)
                    ] = new_label_value
        return new_seg

    def _normalize(self, data, mask):
        """normalize grey values optionally taking into account non-zero maks"""
        data = data.astype(np.float32)
        if not self.cfg["preprocessing"]["norm_mask"]:
            mask = np.ones_like(mask)

        if self.cfg["preprocessing"]["norm_method"] == "minmax":
            # taken from quicknat
            data[mask] = (data[mask] - np.min(data[mask])) / (
                    np.max(data[mask]) - np.min(data[mask])
            )
        elif self.cfg["preprocessing"]["norm_method"] == "std":
            # taken from nnunet
            data[mask] = (data[mask] - data[mask].mean()) / (
                    data[mask].std() + 1e-8
            )
        data[mask == 0] = 0
        return data

    def _clip(self, data, low_thr=-1024, up_thr=600):
        data[data < low_thr] = low_thr
        data[data > up_thr] = up_thr
        return data

