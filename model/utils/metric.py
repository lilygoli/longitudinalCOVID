import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from medpy import metric
from model.utils import metric_utils


def precision(output, target):
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        return precision_score(target, output, average=None)  # average='macro' for macro averaging


def recall(output, target):
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        return recall_score(target, output, average=None)  # average='macro' for macro averaging


def dice_loss(output, target):
    with torch.no_grad():
        return metric_utils.asymmetric_loss(1, output, target)


def dice_score(output, target):
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector

            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        f = f1_score(target, output, average=None)  # average='macro' for macro averaging

        return f


def asymmetric_loss(output, target):
    with torch.no_grad():
        return metric_utils.asymmetric_loss(2, output, target)


lt1, lt2 = [0] * 5, [0] * 5


def LTPR(output, target, is_last=True):
    tprs = []
    global lt1, lt2
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)

        for i in range(5):
            output1 = output.clone()
            target1 = target.clone()

            output1[output1 == i] = 10
            output1[output1 != 10] = 0
            output1[output1 == 10] = 1

            target1[target1 == i] = 10
            target1[target1 != 10] = 0
            target1[target1 == 10] = 1
            output1 = output1.detach().cpu().numpy()
            target1 = target1.detach().cpu().numpy()
            result = np.atleast_1d(output1.astype(np.bool))
            reference = np.atleast_1d(target1.astype(np.bool))

            lt1[i] += np.count_nonzero(result * reference)
            lt2[i] += np.count_nonzero(reference)

            if 0 == lt2[i]:
                tpr = None
            else:
                tpr = lt1[i] / float(lt2[i])

            tprs += [tpr]
    if is_last:
        lt1, lt2 = [0] * 5, [0] * 5
        return tprs
    else:
        return None


lf1, lf2 = [0] * 5, [0] * 5


def LFPR(output, target, is_last=True):
    fprs = []
    global lf1, lf2
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)

        for i in range(5):
            output1 = output.clone()
            target1 = target.clone()

            output1[output1 == i] = 10
            output1[output1 != 10] = 0
            output1[output1 == 10] = 1

            target1[target1 == i] = 10
            target1[target1 != 10] = 0
            target1[target1 == 10] = 1
            output1 = output1.detach().cpu().numpy()
            target1 = target1.detach().cpu().numpy()
            result = np.atleast_1d(output1.astype(np.bool))
            reference = np.atleast_1d(target1.astype(np.bool))

            lf1[i] += np.count_nonzero(result * (1 - reference))
            lf2[i] += np.count_nonzero(reference)

            if 0 == lf2[i]:
                fpr = None
            else:
                fpr = lf1[i] / float(lf2[i])

            fprs += [fpr]
    if is_last:
        lf1, lf2 = [0] * 5, [0] * 5
        return fprs
    else:
        return None


vol1 = [0] * 5
vol2 = [0] * 5


def vd(output, target, is_last=True):
    vds = []
    global vol1, vol2
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)

        for i in range(5):
            output1 = output.clone()
            target1 = target.clone()

            output1[output1 == i] = 10
            output1[output1 != 10] = 0
            output1[output1 == 10] = 1

            target1[target1 == i] = 10
            target1[target1 != 10] = 0
            target1[target1 == 10] = 1
            output1 = output1.detach().cpu().numpy()
            target1 = target1.detach().cpu().numpy()
            result = np.atleast_1d(output1.astype(np.bool))
            reference = np.atleast_1d(target1.astype(np.bool))

            vol1[i] += np.count_nonzero(result)
            vol2[i] += np.count_nonzero(reference)

            vd = abs(vol1[i] - vol2[i])

            vds += [vd]
    if is_last:
        vol1, vol2 = [0] * 5, [0] * 5
        return vds
    else:
        return None
