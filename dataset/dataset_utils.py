import csv
import os
import sys
from collections import defaultdict, OrderedDict
from enum import Enum
from glob import glob
import gc
import h5py
import numpy as np
import pickle
from skimage.transform import resize

from dataset.dynamic.preprocessing import DatasetPreprocessor


class Modalities(Enum):
    SIMPLE = 'simple'


class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'



class Views(Enum):
    SAGITTAL = 0
    CORONAL = 1
    AXIAL = 2


class Mode(Enum):
    STATIC = 'static'
    LONGITUDINAL = 'longitudinal'


class Dataset(Enum):
    ISBI = 'isbi'
    INHOUSE = 'inhouse'


class Evaluate(Enum):
    TRAINING = 'training'
    TEST = 'test'


def retrieve_data_dir_paths(data_dir, evaluate: Evaluate, phase, preprocess, val_patients, mode, size, view=None,
                            sensitivity=False):
    empty_slices, non_positive_slices = None, None

    if preprocess:  # Done only once
        preprocessor = DatasetPreprocessor(
            data_dir + "/config_lung.yml", data_dir, size)  # Place dataset/dynamic/config_lung.yml in a directory next to data

    if preprocess:  # Done only once
        empty_slices, non_positive_slices = preprocess_files(data_dir)

    if mode == Mode.LONGITUDINAL:
        if sensitivity:
            data_dir_paths = retrieve_paths_longitudinal_three(get_patient_paths(data_dir, evaluate, phase),
                                                               phase).items()
        else:
            data_dir_paths = retrieve_paths_longitudinal(get_patient_paths(data_dir, evaluate, phase),
                                                         phase).items()
    else:
        pps = get_patient_paths(data_dir, evaluate, phase)
        data_dir_paths = retrieve_paths_static(pps, phase).items()
    data_dir_paths = OrderedDict(sorted(data_dir_paths))

    _data_dir_paths = []
    patient_keys = [key for key in data_dir_paths.keys()]

    print(patient_keys, flush=True)

    if phase == Phase.TRAIN:
        print(patient_keys, val_patients)
        pk = patient_keys.copy()
        for val_patient in val_patients[::-1]:
            patient_keys.remove(pk[val_patient])

        for patient in patient_keys:
            _data_dir_paths += data_dir_paths[patient]
    elif phase == Phase.VAL:
        for val_patient in val_patients:
            _data_dir_paths += data_dir_paths[patient_keys[val_patient]]
    else:
        for patient in patient_keys:
            _data_dir_paths += data_dir_paths[patient]

    if view:
        _data_dir_paths = list(filter(lambda path: int(path[0].split(os.sep)[-2]) == view.value, _data_dir_paths))
    if phase == Phase.TRAIN or phase == Phase.VAL:
        _data_dir_paths = retrieve_filtered_data_dir_paths(data_dir, phase, _data_dir_paths, empty_slices,
                                                           non_positive_slices,
                                                           mode, val_patients, view)

    return _data_dir_paths


def preprocess_files(root_dir, base_path='data28'):
    patients = list(filter(lambda name: (Evaluate.TRAINING.value) in name,
                           os.listdir(root_dir)))
    empty_slices = []
    non_positive_slices = []
    i_patients = len(patients) + 1

    log_file = [['pat_id_time_step/registered_upon', 0, 1, 2, 3, 4]]
    for patient in patients:

        gc.collect()
        patient_path = os.path.join(root_dir, patient)

        print('Processing patient', patient, flush=True)
        patient_data_path = os.path.join(patient_path, 'preprocessed', patient)
        patient_label_path = os.path.join(patient_path, 'pathMasks', patient)
        patient_lung_path = os.path.join(patient_path, 'masks', patient)

        for modality in list(Modalities):
            mod, value = modality.name, modality.value
            t = 0
            for file in os.listdir(root_dir+f"/{patient}/preprocessed/"):  # change directory
                if file.endswith(".nii"):
                    t += 1

            for i in range(t):  # i registered with j
                for j in range(t):
                    label_path = f'{patient_label_path}_{i + 1}-{j + 1}_pathMask.npy'
                    lung_path = f'{patient_lung_path}_{i + 1}-{j + 1}_mask.npy'
                    data_path = f'{patient_data_path}_{i + 1}-{j + 1}_{value}_pp.npy'
                    normalized_data = np.load(data_path)
                    lung_mask = np.load(lung_path)
                    rotated_labels = np.load(label_path)

                    # create slices through all views
                    path_reg = str(i + 1) + '_' + str(j + 1)

                    log_file += [[patient + path_reg] + (
                            np.histogram(rotated_labels, [0, 1, 2, 3, 4, 5])[0] / (300 ** 3)).tolist()]

                    temp_empty_slices, temp_non_positive_slices = create_slices(normalized_data, rotated_labels,
                                                                                os.path.join(patient_path, base_path,
                                                                                             str(i + 1), path_reg),
                                                                                value)

                    empty_slices += temp_empty_slices
                    non_positive_slices += temp_non_positive_slices

        i_patients += 1
    with open(root_dir + '/hist.csv', 'w') as f:
        csv.writer(f, delimiter=',').writerows(log_file)

    return empty_slices, non_positive_slices


def transform_data(data_path, label, pathology, mins, maxs):  # preprocessing method that doesnt use registration
    data = np.load(data_path)
    if label:
        print("label max:", np.max(data))
        print("label:", np.histogram(data, [0, 1, 2, 3, 4, 5]))
        if not pathology:
            base_mask_path, patient_number = data_path.split('/')[:-2], data_path.split('/')[-1][:-12] + 'mask.npy'
            p = ''
            for jj in base_mask_path:
                p += jj + '/'

            lung_mask = np.load(p + 'masks/' + patient_number)
            data = lung_mask

        if pathology and np.histogram(data, [0, 1, 2, 3, 4, 5])[0][1] == 0:

            base_mask_path, patient_number = data_path.split('/')[:-2], data_path.split('/')[-1][:-12] + 'mask.npy'
            p = ''
            for jj in base_mask_path:
                p += jj + '/'

            lung_mask = np.load(p + 'masks/' + patient_number)

            try:
                data[data + lung_mask == 1] = 1
            except Exception:
                print("EXCEPTION!!!!!! lung mask is None", data.shape, lung_mask.shape)

    out_dims = np.array([300, 300, 300])

    if len(data.shape) == 4:
        data = data[0]

    if label:
        coords = np.argwhere(data)
        # bounding box
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0) + 1  # include top slice

        data = data[x_min:x_max, y_min:y_max, z_min:z_max]
        maxs = [x_max, y_max, z_max]
        mins = [x_min, y_min, z_min]
    else:
        data = data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    x_dim, y_dim, z_dim = data.shape

    x_pad = get_padding(out_dims[0], x_dim)
    y_pad = get_padding(out_dims[1], y_dim)
    z_pad = get_padding(out_dims[2], z_dim)
    smaller = (
        (x_pad[0] < 0) * (out_dims[0] - x_dim), (y_pad[0] < 0) * (out_dims[1] - y_dim),
        (z_pad[0] < 0) * (out_dims[2] - z_dim))
    if smaller != (0, 0, 0):
        new_x, new_y, new_z = data.shape[0] + smaller[0], data.shape[1] + smaller[1], data.shape[2] + smaller[2]
        if label:
            data = resize(data, (new_x, new_y, new_z), order=0)
        else:
            data = resize(data, (new_x, new_y, new_z))

        x_dim, y_dim, z_dim = data.shape
        x_pad = get_padding(out_dims[0], x_dim)
        y_pad = get_padding(out_dims[1], y_dim)
        z_pad = get_padding(out_dims[2], z_dim)
    bigger = (
        (x_pad[0] > 0) * (out_dims[0] - x_dim), (y_pad[0] > 0) * (out_dims[1] - y_dim),
        (z_pad[0] > 0) * (out_dims[2] - z_dim))
    if bigger != (0, 0, 0):
        new_x, new_y, new_z = data.shape[0] + bigger[0], data.shape[1] + bigger[1], data.shape[2] + bigger[2]
        if label:
            data = resize(data, (new_x, new_y, new_z), order=0)
        else:
            data = resize(data, (new_x, new_y, new_z))

        x_dim, y_dim, z_dim = data.shape
        x_pad = get_padding(out_dims[0], x_dim)
        y_pad = get_padding(out_dims[1], y_dim)
        z_pad = get_padding(out_dims[2], z_dim)

    data = np.pad(data, (x_pad, y_pad, z_pad), 'constant')

    return data, mins, maxs


def get_padding(max_dim, current_dim):
    diff = max_dim - current_dim
    pad = diff // 2
    if diff % 2 == 0:
        return pad, pad
    else:
        return pad, pad + 1


def create_slices(data, label, timestep_path, modality):
    empty_slices = []
    non_positive_slices = []
    for view in list(Views):
        name, axis = view.name, view.value
        temp_data = np.moveaxis(data, axis, 0)
        temp_labels = np.moveaxis(label, axis, 0)
        for i, (data_slice, label_slice) in enumerate(zip(temp_data, temp_labels)):
            path = os.path.join(timestep_path, str(axis), f'{i:03}')
            full_path = os.path.join(path, f'{modality}.h5')
            if np.max(data_slice) - np.min(data_slice) <= 1e-5:
                empty_slices.append(path)

            if np.max(label_slice) - np.min(label_slice) <= 1e-5:
                non_positive_slices.append(path)

            while not os.path.exists(full_path):  # sometimes file is not created correctly => Just redo until it exists
                if not os.path.exists(path):
                    os.makedirs(path)
                with h5py.File(full_path, 'w') as data_file:
                    data_file.create_dataset('data', data=data_slice, dtype='f')
                    data_file.create_dataset('label', data=label_slice, dtype='i')
                    data_file.create_dataset('hist', data=label_slice, dtype='i')

    return empty_slices, non_positive_slices


def retrieve_paths_static(patient_paths, phase):
    data_dir_paths = defaultdict(list)

    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        print("patient path", patient_path, flush=True)
        sys.stdout.flush()

        patient = patient_path.split(os.sep)[-2]
        for timestep in filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path)):

            timestep_int = int(timestep)
            timestep_path = os.path.join(patient_path, timestep)

            timestep_path = os.path.join(timestep_path, str(timestep_int) + '_' + str(
                timestep_int))  # in static case we use not registered data bc we dont have reference CT
            if phase != Phase.TRAIN and timestep_int == 1:
                continue
            for axis in filter(lambda x: os.path.isdir(os.path.join(timestep_path, x)), os.listdir(timestep_path)):
                axis_path = os.path.join(timestep_path, axis)
                slice_paths = filter(lambda x: os.path.isdir(x),
                                     map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path)))
                data_dir_paths[patient] += slice_paths

    return data_dir_paths


def retrieve_paths_longitudinal(patient_paths, phase):
    data_dir_paths = defaultdict(list)

    for patient_path in patient_paths:

        if not os.path.isdir(patient_path):
            continue

        patient = patient_path.split(os.sep)[-2]
        for timestep_x in sorted(
                filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
            x_timestep = defaultdict(list)
            timestep_x_int = int(timestep_x)
            timestep_x_path = os.path.join(patient_path, timestep_x)

            timestep_x_path = os.path.join(timestep_x_path, str(timestep_x_int) + '_' + str(timestep_x_int))
            for axis in sorted(
                    filter(lambda x: os.path.isdir(os.path.join(timestep_x_path, x)), os.listdir(timestep_x_path))):
                axis_path = os.path.join(timestep_x_path, axis)
                slice_paths = sorted(filter(lambda x: os.path.isdir(x),
                                            map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                x_timestep[axis] = slice_paths

            for timestep_x_ref in sorted(
                    filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
                x_ref_timestep = defaultdict(list)
                timestep_x_ref_int = int(timestep_x_ref)
                timestep_x_ref_path = os.path.join(patient_path, timestep_x_ref)

                timestep_x_ref_path = os.path.join(timestep_x_ref_path,
                                                   str(timestep_x_ref_int) + '_' + str(
                                                       timestep_x_int))  # here we use reference CT that is registered to target CT
                for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)),
                                          os.listdir(timestep_x_ref_path))):
                    axis_path = os.path.join(timestep_x_ref_path, axis)
                    slice_paths = sorted(filter(lambda x: os.path.isdir(x),
                                                map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                    x_ref_timestep[axis] = slice_paths
                    if phase == Phase.TRAIN:
                        if timestep_x_int != timestep_x_ref_int:  # all_combinations
                            data_dir_paths[patient] += zip(x_ref_timestep[axis], x_timestep[axis])
                    else:
                        if timestep_x_int == timestep_x_ref_int + 1:  # just (ref, target) = (1,2) or (2,3) is sent to trainer and the reverse order (2,1) is used in trainer too
                            data_dir_paths[patient] += zip(x_ref_timestep[axis], x_timestep[axis])
        sys.stdout.flush()
    return data_dir_paths


def retrieve_paths_longitudinal_three(patient_paths,
                                      phase=Phase.VAL):  # to retrieve triples of CTs for patients with three sessions (for sensitivity analysis)
    data_dir_paths = defaultdict(list)

    for patient_path in patient_paths:

        if not os.path.isdir(patient_path):
            continue

        patient = patient_path.split(os.sep)[-2]
        for timestep_x in sorted(
                filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
            x_timestep = defaultdict(list)
            timestep_x_int = int(timestep_x)
            timestep_x_path = os.path.join(patient_path, timestep_x)

            timestep_x_path = os.path.join(timestep_x_path, str(timestep_x_int) + '_' + str(timestep_x_int))
            for axis in sorted(
                    filter(lambda x: os.path.isdir(os.path.join(timestep_x_path, x)), os.listdir(timestep_x_path))):
                axis_path = os.path.join(timestep_x_path, axis)
                slice_paths = sorted(filter(lambda x: os.path.isdir(x),
                                            map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                x_timestep[axis] = slice_paths

            x_ref_timestep = [defaultdict(list), defaultdict(list)]
            i = -1
            for timestep_x_ref in sorted(
                    filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
                timestep_x_ref_int = int(timestep_x_ref)
                timestep_x_ref_path = os.path.join(patient_path, timestep_x_ref)

                if timestep_x_int != timestep_x_ref_int:
                    i += 1
                else:
                    continue
                timestep_x_ref_path = os.path.join(timestep_x_ref_path,
                                                   str(timestep_x_ref_int) + '_' + str(timestep_x_int))
                for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)),
                                          os.listdir(timestep_x_ref_path))):
                    axis_path = os.path.join(timestep_x_ref_path, axis)
                    slice_paths = sorted(filter(lambda x: os.path.isdir(x),
                                                map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                    x_ref_timestep[i][axis] = slice_paths

            if i < 1:
                continue

            for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)),
                                      os.listdir(timestep_x_ref_path))):
                data_dir_paths[patient] += zip(x_ref_timestep[0][axis], x_ref_timestep[1][axis], x_timestep[axis])

        sys.stdout.flush()
    return data_dir_paths


def get_patient_paths(data_dir, evaluate, phase):
    patient_paths = map(lambda name: os.path.join(name, 'data28'),  # change directory
                        (filter(
                            lambda name: (evaluate.value if phase == Phase.VAL else Evaluate.TRAINING.value) in name,
                            glob(os.path.join(data_dir, '*')))))

    return patient_paths


def retrieve_filtered_data_dir_paths(root_dir, phase, data_dir_paths, empty_slices, non_positive_slices, mode,
                                     val_patients, view: Views = None):
    empty_file_path = os.path.join(root_dir, 'empty_slices28.pckl')
    non_positive_slices_path = os.path.join(root_dir, 'non_positive_slices28.pckl')

    if empty_slices:
        pickle.dump(empty_slices, open(empty_file_path, 'wb'))
    if non_positive_slices:
        pickle.dump(non_positive_slices, open(non_positive_slices_path, 'wb'))

    data_dir_path = os.path.join(root_dir,
                                 f'data_dir_{mode.value}_{phase.value}_{val_patients}{f"_{view.name}" if view else ""}28.pckl')
    if os.path.exists(data_dir_path):
        # means it has been preprocessed before -> directly load data_dir_paths
        data_dir_paths = pickle.load(open(data_dir_path, 'rb'))
        print(f'Elements in data_dir_paths: {len(data_dir_paths)}')
    else:
        if not empty_slices:
            empty_slices = pickle.load(open(empty_file_path, 'rb'))
        if not non_positive_slices:
            non_positive_slices = pickle.load(open(non_positive_slices_path, 'rb'))
        print(f'Elements in data_dir_paths before filtering empty slices: {len(data_dir_paths)}')
        if mode == Mode.STATIC:
            data_dir_paths = [x for x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]
        else:
            data_dir_pathss = []
            for x_ref_1, x in data_dir_paths:
                if x not in set(empty_slices + non_positive_slices) and phase != Phase.TEST:
                    data_dir_pathss += [(x_ref_1, x)]  # ,(x_ref_1, x) add for augmentation (doubling data)
                else:
                    data_dir_pathss += [(x_ref_1, x)]

            data_dir_paths = data_dir_pathss
        print(f'Elements in data_dir_paths after filtering empty slices: {len(data_dir_paths)}')
        pickle.dump(data_dir_paths, open(data_dir_path, 'wb'))

    return data_dir_paths

