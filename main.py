import argparse
import os
import random
from collections import defaultdict
from copy import copy

import numpy as np
import torch

import data_loader as module_data_loader
import dataset as module_dataset
import model as module_arch
import model.utils.loss as module_loss
import model.utils.metric as module_metric
import trainer as trainer_module
from dataset.ISBIDatasetStatic import Phase
from dataset.dataset_utils import Views
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume=None):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    num_patients = config['dataset']['num_patients']
    crossVal_or_test = False
    if config['test']:
        folds = 1
        len_fold = num_patients
        crossVal_or_test = True

    elif config['dataset']['cross_val']:
        folds = config['dataset']['val_fold_num']
        len_fold = config['dataset']['val_fold_len']
        crossVal_or_test = True

    else:
        folds, len_fold = 1, 0
        if config['dataset']['args']['val_patients']:
            raise Exception(
                "Please specify validation patients set in config while not using cross-validation or test phase.")


    all_patients = [i for i in range(num_patients)]
    np.random.shuffle(all_patients)
    if resume:
        config.resume = resume

    logger = config.get_logger('train')

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # setup data_loader instances
    if config['single_view']:

        results = defaultdict(list)
        for view in list(Views):
            _cfg = copy(config)
            for fold in range(folds):
                logger.info('Fold Number: {}'.format(fold + 1))
                logs = train(logger, _cfg, loss, metrics, fold, len_fold, all_patients, crossVal_or_test, view=view)
                for k, v in list(logs.items()):
                    results[k].append(v)

    else:
        for fold in range(folds):
            logger.info('Fold Number: {}'.format(fold + 1))
            train(logger, config, loss, metrics, fold, len_fold, all_patients, crossVal_or_test)


def train(logger, config, loss, metrics, fold, len_fold, all_patients, crossVal_or_test, view: Views = None):
    logger.info('start trainning: {}'.format(config['dataset']['args']))
    print("Cross of test", crossVal_or_test, all_patients, fold, len_fold, flush=True)
    if crossVal_or_test:
        config['dataset']['args']['val_patients'] = all_patients[fold * len_fold: (fold + 1) * len_fold]

    data_loader = None
    if len(all_patients) != len(config['dataset']['args']['val_patients']):  # if we had any patients left in the train set
        dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.TRAIN,
                                                                   view=view)

        data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'],
                                                                               dataset=dataset)

    val_dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.VAL,
                                                                   view=view)

    valid_data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'],
                                                                                 dataset=val_dataset)
    # build model architecture, then print to console
    model = config.initialize_class('arch', module_arch)


    logger.info(model)
    if config['only_validation'] or config['test']:
        logger.info('Loading checkpoint: {} ...'.format(config['path']))
        path = config["path"]

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)


    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if view:
        config._save_dir = os.path.join(config._save_dir, str(view.name))
        config._log_dir = os.path.join(config._log_dir, str(view.name))
        os.mkdir(config._save_dir)
        os.mkdir(config._log_dir)

    trainer = config.retrieve_class('trainer', trainer_module)(model, loss, metrics, optimizer, config, data_loader,
                                                               fold, valid_data_loader, lr_scheduler)

    return trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--single_view', default=False, type=bool,
                      help='Defines if a single is used per plane orientation')
    args.add_argument('-v', '--only_validation', default=False, type=bool,
                      help='just run validation on a checkpoint model and do no training -- should add argument -p')
    args.add_argument('-p', '--path', default=None, type=str, help='path to latest checkpoint (default: None)')

    args.add_argument('-t', '--test', default=False, type=bool,
                      help='to run test phase on all the patients list')

    config = ConfigParser(*parse_cmd_args(args))
    main(config)
