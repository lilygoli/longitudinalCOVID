import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def visualize_flow(flow):
    """Visualize optical flow

    Args:
        flow: optical flow map with shape of (H, W, 2), with (y, x) order

    Returns:
        RGB image of shape (H, W, 3)
    """
    assert flow.ndim == 3
    assert flow.shape[2] == 2

    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 1], flow[..., 0])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_difference(x):
    rgbs = []
    for i in x:
        hsv = np.zeros([i.shape[0], i.shape[1], 3], dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        hsv[..., 0] = i * 255 // 2  # cv2.normalize(i, None, 0, 255, cv2.NORM_INF)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgbs += [rgb]
    return np.array(rgbs)




def prepare_encoded(encoded):
    encoded = encoded.detach().cpu().numpy().astype('float32')
    heatmap = np.mean(encoded, axis=1).squeeze()    # mean on the channels

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= np.max(heatmap)
    return heatmap


def log_visualizations(writer, x_ref, x, output, target, output_ref, target_ref, outputDiff,
                       groundDiff, outputDiffReverse, groundDiffReverse, encoded, toy=False):
    batch_size = x.size(0)
    x_ref = cast(x_ref)
    x = cast(x)
    output = cast(output, True, True, True)
    target = cast(target, True, True, True)
    if output_ref is not None and groundDiff is not None:

        outputDiff = visualize_difference(outputDiff.cpu().detach().numpy()).astype('float32')
        groundDiff = visualize_difference(groundDiff.cpu().detach().numpy()).astype("float32")
        outputDiffReverse = visualize_difference(outputDiffReverse.cpu().detach().numpy()).astype('float32')
        groundDiffReverse = visualize_difference(groundDiffReverse.cpu().detach().numpy()).astype("float32")

        output_ref = cast(output_ref, True, True, True)
        target_ref = cast(target_ref, True, True, True)

        if encoded is not None:
            encoded = np.reshape(np.array([prepare_encoded(encoded)]), (batch_size, 9, 9, 1))
            for i in range(batch_size):
                if not toy:
                    a1, a2, b1, b, c, c1 = x_ref[i], x[i], output_ref[i], target_ref[i], output[i], target[i]

                    tensor1 = np.expand_dims(np.transpose(np.hstack([a1, a2, b1, b, c, c1]), (2, 0, 1)), axis=0)
                    writer.add_image('xRef_x_outputRef_targetRef_output_target',
                                     make_grid(torch.as_tensor(tensor1), nrow=8, normalize=False))


                else:
                    a, a1, b, b1, c, c1 = x[i], x_ref[i], output_ref[i], target_ref[i], output[i], target[i]
                    tensor2 = np.expand_dims(np.transpose(np.hstack([a, a1, b, b1, c, c1]), (2, 0, 1)), axis=0)
                    writer.add_image('TOY_x_xref_outputRef_targetRef_output_target',
                                     make_grid(torch.as_tensor(tensor2), nrow=8, normalize=False))

                if not toy:
                    d, e, f, g = outputDiff[i], groundDiff[i], outputDiffReverse[i], groundDiffReverse[i]
                    tensor3 = np.expand_dims(np.transpose(np.hstack([d, e, f, g]), (2, 0, 1)), axis=0)
                    writer.add_image('outDiff_groundDiff_outDiffReverse_groundDiffReverse',
                                     make_grid(torch.as_tensor(tensor3), nrow=8, normalize=False))


                else:
                    d, e, f, g = outputDiff[i], groundDiff[i], outputDiffReverse[i], groundDiffReverse[i]
                    tensor4 = np.expand_dims(np.transpose(np.hstack([d, e, f, g]), (2, 0, 1)), axis=0)
                    writer.add_image('TOY_outDiff_groundDiff_outDiffReverse_groundDiffReverse',
                                     make_grid(torch.as_tensor(tensor4), nrow=100, normalize=False))

                if encoded is not None:
                    if not toy:
                        encodedd = encoded[i]
                        tensor5 = np.expand_dims(np.transpose(encodedd, (2, 0, 1)), axis=0)
                        writer.add_image('encodedLongitudinal',
                                         make_grid(torch.as_tensor(tensor5), nrow=8, normalize=False))
                    else:
                        x_toy = encoded[i]
                        tensor5 = np.expand_dims(np.transpose(x_toy, (2, 0, 1)), axis=0)
                        writer.add_image('encodedTOY',
                                         make_grid(torch.as_tensor(tensor5), nrow=8, normalize=False))
    elif groundDiff is None and output_ref is not None:
        for i in range(batch_size):
            a1, a2, b, b1, c, c1 = x_ref[i], x[i], output_ref[i], target_ref[i], output[i], target[i]
            tensor = np.expand_dims(np.transpose(np.hstack([a1, a2, b, b1, c, c1]), (2, 0, 1)), axis=0)
            writer.add_image('xRef_x_outputRef(2)_targetRef_output_target',
                             make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))
    else:
        for i in range(batch_size):
            a1, a2, b, c = x_ref[i], x[i], output[i], target[i]
            tensor = np.expand_dims(np.transpose(np.hstack([a1, a2, b, c]), (2, 0, 1)), axis=0)
            writer.add_image('xRef_x_output_target',
                             make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))


def log_visualizations_deformations(writer, input_moving, input_fixed, flow, target_moving, target_fixed, output=None):
    zipped_data = zip(
        cast(input_moving),
        cast(input_fixed),
        cast(flow, normalize_data=False),
        cast(target_moving, True),
        cast(target_fixed, True),
        cast(output, True) if type(None) != type(output) else [None for _ in input_moving]
    )
    for (_input_moving, _input_fixed, _flow, _target_moving, _target_fixed, _output) in zipped_data:
        transposed_flow = np.transpose(_flow, (1, 2, 0))

        illustration = [
            _input_moving,
            _input_fixed,
            visualize_flow(transposed_flow) / 255.,
            _target_moving,
            _target_fixed
        ]
        if type(None) != type(_output):
            illustration.append(_output)

        tensor = np.expand_dims(np.transpose(np.hstack(illustration), (2, 0, 1)), axis=0)
        description = 'xRef_x_flowfield_targetRef_target_output'
        writer.add_image(description, make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))


def cast(data, argmax=False, normalize_data=True, mask=False):
    data2 = data.cpu().detach().numpy()
    if argmax:
        data2 = np.argmax(data2, axis=1)

    data2 = data2.astype('float32')

    if normalize_data:
        data2 = np.asarray([normalize(date, mask) for date in data2])

    return data2


def normalize(x, mask):

    if len(x.shape) > 2:
        x = x[0]
    if mask:
        hsv = np.zeros([x.shape[0], x.shape[1], 3], dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        hsv[..., 0] = x * 255 // 4
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # cv2.cvtColor(x * 1/4, cv2.COLOR_GRAY2RGB) #cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX) for grayscale
        return rgb
    else:
        return cv2.cvtColor(cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX), cv2.COLOR_GRAY2RGB)
