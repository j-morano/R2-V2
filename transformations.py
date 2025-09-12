from typing import List, Union

import numpy as np
import torch



def get_unet_padding_np(np_image: np.ndarray, n_down=5) -> tuple:
    """ Calculates the necessary padding of an image to be processed by
    UNet.

    Args:
        np_image: image in NumPy format.
        n_down: number of downsampling blocks.

    Returns:
        Image padding (NumPy format).
    """
    n = 2**n_down
    shape = np_image.shape
    h_pad = n - shape[0] % n
    w_pad = n - shape[1] % n
    h_half_pad = int(h_pad/2)
    w_half_pad = int(w_pad/2)
    if len(shape) == 3:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad), (0, 0)
    else:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad)
    return padding


def pad_images_unet(
    np_images: List[np.ndarray],
    return_paddings: bool=False,
) -> Union[tuple, list]:
    """ Applies UNet padding to a list of images in NumPy format.

    Args:
        np_images: list of NumPy images.

    Returns:
        Padded images.
    """
    padded_images = []
    paddings = []
    for np_image in np_images:
        padding = get_unet_padding_np(np_image)
        paddings.append(padding)
        padded_images.append(np.pad(np_image, padding))
    if return_paddings:
        return padded_images, paddings
    else:
        return padded_images


def to_torch_tensors(npimages, device=None):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    tensors = []
    for img in npimages:
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        tensor = torch.from_numpy(img.transpose(2, 0, 1).astype('float32'))
        if device is not None:
            tensor = tensor.to(device)
        tensors.append(tensor.unsqueeze(0))
    return tensors
