from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import hashlib
import torch


def load_config_yaml(path):
    """loads a yaml config from file and returns a dict"""
    path = Path(path)
    with open(path) as file:
        cfg = yaml.full_load(file)
    return cfg


def save_config_yaml(path, config):
    path = Path(path)
    with open(path, "w") as file:
        yaml.dump(config, file)


def split_idxs(idxs_in, test_size=0.1, val_size=0.1, seed=42, shuffle=True):
    """split indices into test, val and train
    """
    idxs_out = {}
    if test_size > 0:
        idxs_out["train"], idxs_out["test"] = train_test_split(
            idxs_in, test_size=test_size, shuffle=shuffle, stratify=None, random_state=seed
        )
    else:
        idxs_out["test"] = []
        idxs_out["train"] = idxs_in
    if val_size > 0:
        idxs_out["train"], idxs_out["val"] = train_test_split(
            idxs_out["train"],
            test_size=val_size / (1 - test_size),
            shuffle=True,
            stratify=None,
            random_state=seed,
        )
    else:
        idxs_out["val"] = []
    return idxs_out


def rm_tree(pth: Path):
    """WARNING: deletes path recursively like rm -rf"""
    print(f"Recursively deleting '{pth}'")
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def get_sha256_hash(path):
    """returns sha256 hash from file found at path"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_hash(hash, path):
    """save hash to given path"""
    with open(path, "w") as hash_file:
        print(hash, file=hash_file, end="")


def load_hash(path):
    """load hash from path"""
    with open(path, "r") as hash_file:
        return hash_file.read()


def verify_config_hash(config_path, npy_path: Path):
    """checks if config is the same as hashed and return bool"""
    hash_path = npy_path / "config_hash.sha256"
    if hash_path.is_file():
        new_hash = get_sha256_hash(config_path)
        old_hash = load_hash(hash_path)
        if new_hash == old_hash:
            return True
    return False


def save_config_hash(config_path, npy_path: Path):
    """saves hash of given config"""
    cfg_hash = get_sha256_hash(config_path)
    hash_path = npy_path / "config_hash.sha256"
    save_hash(cfg_hash, hash_path)


def make_config(cfg, dyndata_path):
    """write a config yaml file based on the cfg dictionary provided"""
    pp_path = dyndata_path
    setup_yml_path = pp_path / "setup.yml"
    assert Path(
        setup_yml_path
    ).is_file(), f"setup yaml could not be found at '{setup_yml_path}'"
    setup = load_config_yaml(setup_yml_path)
    cfg["setup_hash"] = get_sha256_hash(setup_yml_path)

    if "labels" not in cfg.keys():
        assert (
                "labelmap" in cfg.keys()
        ), "labelmap needs to be specified check setup script"
        labels_dict = setup["labels"][cfg["labelmap"]]
        cfg["labels"] = sorted(labels_dict, key=labels_dict.get)

    cfg_path = (pp_path / f"config_{cfg['name']}.yml").absolute()
    save_config_yaml(cfg_path, cfg)
    print(
        f"'{cfg['name']}' config for '{setup['name']}' dataset \nwas successfully saved to '{cfg_path}'"
    )


def to_crop_padded_tensor_3d(data, out_dims=[64, 64, 64], padding_value=0):
    """ pads a list of numpy arrays to given output dimension and
    returns one big tensor """
    num_chan = data.shape[0]
    data = torch.from_numpy(data)
    out_shape = [num_chan, *out_dims]
    out_dims = torch.tensor(out_dims)
    out_tensor = torch.full(size=out_shape, fill_value=padding_value, dtype=data.dtype)

    for i in range(num_chan):
        in_dims = torch.tensor(data[i].shape)
        padding = (out_dims - in_dims) / 2
        start = padding.clone()
        start_data = torch.zeros_like(padding)
        end_data = in_dims.clone()
        end = padding + in_dims

        # check if tensor needs to be cropped
        for d in range(3):
            if in_dims[d] > out_dims[d]:
                start[d] = 0
                start_data[d] = -padding[d]
                end[d] = out_dims[d]
                end_data[d] = start_data[d] + out_dims[d]

        out_tensor[
        i, start[0]:end[0], start[1]:end[1], start[2]:end[2]
        ] = data[i, start_data[0]:end_data[0], start_data[1]:end_data[1], start_data[2]:end_data[2]]
    return out_tensor


def random_narrow_tensor(tensors, narrow_size, dim=0, include="center", ignore_bg=True):
    non_zero = (
            tensors[1][ignore_bg:] != 0
    ).nonzero()  # Contains non-zero indices for all 4 dims
    h_min = non_zero[:, dim].min()
    h_max = non_zero[:, dim].max()
    if include == "target":
        start_slice = int(
            np.clip(
                (h_min + (((h_max - h_min) - narrow_size)) * np.random.random()),
                0,
                tensors[0].size(dim) - narrow_size,
            )
        )
    elif include == "center":
        start_slice = int(
            np.clip(
                ((h_min + (h_max - h_min) / 2) - narrow_size / 2),
                0,
                tensors[0].size(dim) - narrow_size,
            )
        )
    elif include == "random":
        start_slice = np.random.randint(tensors[0].size(dim) - narrow_size)
    else:
        return tensors
    for i in range(len(tensors)):
        tensors[i] = torch.narrow(tensors[i], dim, start_slice, narrow_size)
    return tensors


def crop_to_mask(data, seg,
                 lung):  # crops segmentation mask and data to where the lung mask (and segmentation) mask are non-zero
    """
        crop data and return non-zero mask
        inspired by nnunet and stackoverflow

        # """

    crop_threshold = -1000000000
    mask = np.zeros(data.shape, dtype=bool)
    # non zero mask over all channels

    cmask = data > crop_threshold
    mask = cmask | mask
    # non black coordinates
    coords = np.argwhere(mask)
    # bounding box
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1  # include top slice

    cropped_data = data[x_min:x_max, y_min:y_max, z_min:z_max]

    cropped_seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_lung = lung[x_min:x_max, y_min:y_max, z_min:z_max]
    mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]

    coords = np.argwhere(cropped_seg)
    coords2 = np.argwhere(cropped_lung)
    # bounding box

    x_min, y_min, z_min = np.concatenate((np.array([coords2.min(axis=0)]),np.array([coords.min(axis=0)])), axis=0).min(axis=0)   # change to : 'coords2.min(axis=0)' for only considering lung mask
    x_max, y_max, z_max = np.concatenate((np.array([coords2.max(axis=0)]),np.array([coords.max(axis=0)])), axis=0).max(axis=0) + 1  # include top slice # change to: 'coords2.max(axis=0)' for only considering lung mask

    cropped_lung = cropped_lung[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_seg = cropped_seg[x_min:x_max, y_min:y_max, z_min:z_max]

    cropped_data = cropped_data[x_min:x_max, y_min:y_max, z_min:z_max]

    return np.array(cropped_data), np.array(cropped_seg), np.array(cropped_lung), mask
