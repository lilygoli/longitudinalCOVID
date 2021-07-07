import argparse
import os

import nibabel
import numpy as np
import torch
from scipy.ndimage import rotate
from tqdm import tqdm

import data_loader as module_data_loader
import dataset as module_dataset
import model as module_arch
import model.utils.metric as module_metric
from dataset.ISBIDatasetStatic import Phase
from dataset.dataset_utils import Evaluate, Dataset
from parse_config import ConfigParser, parse_cmd_args

'''For Majority Voting and taking mean over all planes'''


def main(config, resume=None):
    if config["path"]:
        resume = config["path"]

    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.retrieve_class('dataset', module_dataset)(
        **config['dataset']['args'], phase=Phase.TEST, evaluate=config['evaluate']
    )
    data_loader = config.retrieve_class('data_loader', module_data_loader)(
        dataset=dataset,
        batch_size=config['data_loader']['args']['batch_size'],
        num_workers=config['data_loader']['args']['num_workers'],
        shuffle=False
    )

    # build model architecture
    model = config.initialize_class('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    res = config['dataset']['args']['size']

    total_metrics = torch.zeros(len(metric_fns), config['dataset']['args']['n_classes'])
    volume_metrics = torch.zeros(len(metric_fns),config['dataset']['args']['n_classes'])

    with torch.no_grad():
        # setup
        volume = 0
        axis = 0  # max 2
        c = 0
        alignment = [(0, 1, 2), (1, 0, 2), (2, 1, 0)]
        data_shape = [res, res, res]

        output_agg = torch.zeros([config['dataset']['args']['n_classes'], *data_shape]).to(device)
        # avg_seg_volume = None
        target_agg = torch.zeros([config['dataset']['args']['n_classes'], *data_shape]).to(device)

        n_samples = 0
        for idx, loaded_data in enumerate(tqdm(data_loader)):
            if len(loaded_data) == 6:
                # static case

                data, target = loaded_data[0], loaded_data[1]
                data, target = data.to(device), target.to(device)
                output = model(data)
            else:
                # longitudinal case

                x_ref, x, _, target = loaded_data[0], loaded_data[1], loaded_data[2], loaded_data[3]
                x_ref, x, target = x_ref.to(device), x.to(device), target.to(device)
                output,_ = model(x_ref, x)
               


            for cl in range(output_agg.size()[0]):
                x = output_agg[cl].to('cpu').numpy()
                y = output[0][cl].to('cpu').numpy()
                z = np.transpose(x, alignment[axis])
                z[c] += y
                output_agg[cl] = torch.tensor(np.transpose(z, alignment[axis])).to(device)
            for cl in range(output_agg.size()[0]):
                x = target_agg[cl].to('cpu').numpy()
                y = target[0][cl].to('cpu').numpy()
                z = np.transpose(x, alignment[axis])
                z[c] += y
                target_agg[cl] = torch.tensor(np.transpose(z, alignment[axis])).to(device)



            c += 1
            print("C is: ", c, "res is: ", res, flush=True)
            if c == res:
                axis += 1
                c = 0
                print("Axis Changed ", axis)
                if axis == 3:
                    print("Volume finished")
                    path = os.path.join(config.config['trainer']['save_dir'], 'output',
                                        *str(config._save_dir).split(os.sep)[-2:],
                                        str(resume).split(os.sep)[-1][:-4])

                    os.makedirs(path, exist_ok=True)
                    axis = 0

                    label_out = output_agg.argmax(0)
                    label_target = target_agg.argmax(0)

                    evaluate_timestep(output_agg.unsqueeze(0), target_agg.unsqueeze(0), label_out, label_target, metric_fns, config, path, volume,
                                      volume_metrics, total_metrics,
                                      logger)

                    # inferred whole volume
                    logger.info('---------------------------------')
                    logger.info(f'Volume number {int(volume) + 1}:')
                    for i, met in enumerate(metric_fns):
                        logger.info(f'      {met.__name__}: {volume_metrics[i]}')
                    volume_metrics = torch.zeros(len(metric_fns))

                    volume += 1


    logger.info('================================')
    logger.info(f'Averaged over all patients:')
    for i, met in enumerate(metric_fns):
        logger.info(f'      {met.__name__}: {total_metrics[i].item() / n_samples}')


def evaluate_timestep(avg_seg_volume, target_agg, label_out, label_target, metric_fns, config, path, patient, volume_metrics, total_metrics,
                      logger):

    prefix = f'{config["evaluate"].value}{(int(patient) + 1):02}'
    seg_volume = label_out.int().cpu().detach().numpy()
    rotated_seg_volume = rotate(rotate(seg_volume, -90, axes=(0, 1)), 90, axes=(1, 2))

    nibabel.save(nibabel.Nifti1Image(rotated_seg_volume, np.eye(4)), os.path.join(path, f'{prefix}_seg.nii'))

    target_volume = label_target.int().cpu().detach().numpy()
    rotated_target_volume = rotate(rotate(target_volume, -90, axes=(0, 1)), 90, axes=(1, 2))
    nibabel.save(nibabel.Nifti1Image(rotated_target_volume, np.eye(4)), os.path.join(path, f'{prefix}_target.nii'))
    # computing loss, metrics on test set
    logger.info(f'Patient {int(patient) + 1}: ')
    for i, metric in enumerate(metric_fns):
        if metric.__name__.__contains__("loss"):
            continue
        current_metric = metric(avg_seg_volume, target_agg)
        logger.info(f'      {metric.__name__}: {current_metric}')
        try:
            for j in range(current_metric.shape[0]):
                volume_metrics[i][j] += current_metric[j]
                total_metrics[i][j] += current_metric[j]
        except Exception:
            print("Invalid metric shape.")
            continue




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--evaluate', default=Evaluate.TEST, type=Evaluate,
                      help='Either "training" or "test"; Determines the prefix of the folders to use')
    args.add_argument('-m', '--dataset_type', default=Dataset.ISBI, type=Dataset, help='Dataset to use')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-p', '--path', default=None, type=str, help='path to latest checkpoint (default: None)')
    config = ConfigParser(*parse_cmd_args(args))
    main(config)

