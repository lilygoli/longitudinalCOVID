import os
from polyaxon_client.tracking import get_outputs_path


CONFIG = {
    "name": f"{os.path.basename(__file__).split('.')[0]}",
    "n_gpu": 1,

    "arch": {
        "type": "LongitudinalFCDenseNet",
        "args": {
            "in_channels": 1,
            "siamese": False,
            "n_classes": 5
        }
    },
    "dataset": {
        "type": "DatasetLongitudinal",
        "num_patients": 22,
        "cross_val": True,
        "val_fold_num": 4,
        "val_fold_len": 4,
        "args": {
            "data_dir": "/data/COVID_longitudinal/test",
            "preprocess": False,
            "size": 300,
            "n_classes": 5,
            "modalities": ['simple'],
            "val_patients": None  # if not using cross validation: [2,6,10,14] or an arbitrary array of patient numbers
        }
    },
    "data_loader": {
        "type": "Dataloader",
        "args": {
            "batch_size": 2,
            "shuffle": True, # for test to use LTPR and VD metrics use False
            "num_workers": 4,
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.0001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "mse",
    "metrics": [
        "precision", "recall", "dice_loss", "dice_score", "asymmetric_loss"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "type": "LongitudinalMaskPropagationTrainer",
        "epochs": 1, #change to 100 in Train Phase
        "save_dir": get_outputs_path(),
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": True
    }
}
