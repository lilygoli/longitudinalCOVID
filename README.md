# Longitudinal Quantitative Assessment of COVID-19 Infection Progression from Chest CTs

This is the code for our paper Longitudinal Quantitative Assessment of COVID-19 Infection Progression from Chest CTs which can be found [here](https://arxiv.org/pdf/2103.07240.pdf)

If you use any of our code, please cite:
```
@article{kim2021longitudinal,
    title={Longitudinal Quantitative Assessment of COVID-19 Infection Progression from Chest CTs},
    author={Seong Tae Kim and Leili Goli and Magdalini Paschali and Ashkan Khakzar and Matthias Keicher and Tobias Czempiel and Egon Burian and Rickmer Braren and Nassir Navab and Thomas Wendler},
    year={2021},
    eprint={2103.07240},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
``` 


* [Longitudinal Quantitative Assessment of COVID-19 Infection Progression from Chest CTs Segmentation](#Longitudinal-Quantitative-Assessment-of-COVID-19-Infection-Progression-from-Chest-CTs)
    * [Requirements](#requirements)
    * [Folder Structure](#folder-structure)
    * [Usage](#usage)
        * [Train](#train)
        * [Validation and Cross Validation](#validation)
        * [Test](#test)


## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch = 1.4 
* tqdm
* tensorboard >= 1.14 
* nibabel >= 2.5

## Folder Structure
  ```
  longitudinal-covid-19-network/
  │
  ├── main.py - main script to start/resume training or do test 
  │
  ├── base/ - abstract base classes
  │  
  ├── configs/ - holds all the configurations files for the different models
  │   ├── Longitudinal_Network.py
  │   ├── Longitudinal_Late_Fusion.py
  │   ├── Longitudinal_Network_with_Progression_Learning.py
  │   └── Static_Network.py
  │
  ├── data_loader/
  │   └── ISBIDataloader.py - dataloader for the ISBI Dataset
  │
  ├── model/
  │   ├── utils/ - holds additional Modules, losses and metrics
  │   ├── FCDenseNet.py
  │   ├── LongitudinalFCDenseNet.py
  │   └── LateLongitudinalFCDenseNet.py
  │
  └── trainer/ - trainers
      ├── ISBITrainer.py
      ├── LongitudinalTrainer.py
      ├── LongitudinalWithProgressionTrainer.py
      └── StaticTrainer.py

  ```

## Usage
Before the models can be trained or tested, the paths in the config files (located in `configs/`) have to be adjusted:
- `data_loader.args.data_dir` specifies where the data is located
-  `trainer.save_dir` specifies where to store the model checkpoints and logs.
-  `dataset.num_patients` specifies number of patient files that should be processed.
-  `dataset.val_fold_num` specifies number of validation folds in cross validation setting.
-  `dataset.val_fold_len` specifies length of validation fold in terms of number of patients.


### Train
To run the experiments from our paper the following table specifies the commands to run:

| Network                                   | Command                                                              |
|-------------------------------------------|----------------------------------------------------------------------|
| Longitudinal Network            | python main.py -c Longitudinal_Network.py                 |
| Longitudinal Network with Progression loss     | python main.py -c Longitudinal_Network_with_Progression_Learning.py |
| Longitudinal Late Fusion Network  (multi-view)                   | python main.py -c Longitudinal_Late_Fusion.py                           |
| Static Network                            | python main.py -c Static_Network.py                                 |

### Validation and Cross Validation

For performing Cross validation set cross validation to True in the config file of the network that is going to be used. Also set number of folds and fold length in the config file.

Specific set of patients can also be specified in config files to be used as validation set. 

To only run validation or cross validation on previously trained models (without training again) use the format below:
  ```
  python main.py -c <config .py for the network>  -v True --path <path to model>
  ```
example:
  ```
  python main.py -c Static_Network.py  -v True --path
    /data/COVID_longitudinal/best/static/model.pth
  ```

### Test

Test a model by executing the following format:
  ```
  python main.py -c <config .py for the network>  -t True --path <path to model>
  ```
example:

  ```
  python main.py -c Static_Network.py  -t True --path
    /data/COVID_longitudinal/best/static/model.pth
  ```
Also the Majority Voting code can be executed by using the following format:

  ```
  python main.py -c <config .py for the network>  -t True --path <path to model>
  ```
example:

  ```
  python MajorityVoting.py -c Static_Network.py  -e test --path
    /data/COVID_longitudinal/best/static/model.pth
  ```

