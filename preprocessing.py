from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import skimage.io as io
from skimage.exposure import equalize_adapthist
from scipy.interpolate import interp1d
from skimage.morphology import disk



def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def to_0_1(img):
    interp_fun = interp1d([img.min(), img.max()], [0.0, 1.0])
    return interp_fun(img)


def enhance_image(img, mask, int_format=False, disk_size=5):
    """Enhance an image using the method described in the paper.
    Args:
        img (uint8 np.ndarray): Image to enhance
        mask (np.ndarray): ROI mask of the image to enhance

    Returns:
        Enhanced image
    """
    # Read image and its corresponding mask
    if isinstance(img, str) or isinstance(img, Path):
        img = io.imread(img)[..., :3]
    if isinstance(mask, str) or isinstance(mask, Path):
        mask = io.imread(mask)

    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    # mask = np.where(mask > (255//2), 255, 0)
    mask = np.where(mask > 0.5, 1, 0)

    # Copy original image
    img_copy = img.copy()
    # Convert to PIL format
    zoomed_image = Image.fromarray(np.uint8(img_copy*255))
    # Enlarge image
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1]*1.15), int(img_copy.shape[0]*1.15)),
        Image.BICUBIC  # type: ignore
    )
    # To numpy array type
    zoomed_image = np.array(zoomed_image)
    # Crop image to original size (zoom result)
    zoomed_image = crop_center(zoomed_image, img_copy.shape[1],
                               img_copy.shape[0])
    # Convert image from 0-255 format to 0.0-1.0 format
    zoomed_image = zoomed_image / 255.0

    # Create circular kernel for mask erosion
    kernel = disk(disk_size)

    # Erode mask
    mask = ndimage.binary_erosion(mask, kernel)
    # Convert boolean array to float array
    mask = mask * 1.0  # type: ignore

    img_copy[mask < 1.0] = 0.0

    # Create RGB mask (same mask for all channels)
    mask = np.stack((mask, mask, mask), axis=2)

    composed_image = mask.copy()
    composed_image[mask == 1.0] = img_copy[mask == 1.0]
    composed_image[mask < 1.0] = zoomed_image[mask < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask < 1.] = 0.

    enhanced_image = subtracted_image/np.std(subtracted_image)
    enhanced_image = to_0_1(enhanced_image)
    enhanced_image[mask < 1.] = 0.

    mask = mask[:, :, 0]

    if int_format:
        enhanced_image *= 255
        enhanced_image = enhanced_image.astype(np.uint8)
        mask *= 255
        mask = mask.astype(np.uint8)

    return enhanced_image, mask


def preprocess_img(img, mask):
    img_enh, mask_enh = enhance_image(img, mask, int_format=True)
    img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
    return img_enh_clahe, mask_enh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess images and masks.')
    parser.add_argument(
        '-c', '--cfp_path',
        type=str,
        required=True,
        help='Path to the images to preprocess',
    )
    parser.add_argument(
        '-m', '--mask_path',
        type=str,
        required=True,
        help='Path to the masks to preprocess',
    )
    parser.add_argument(
        '-p', '--out_pre_path',
        type=str,
        required=True,
        help='Path to save the preprocessed images',
    )
    parser.add_argument(
        '-o', '--out_mask_path',
        type=str,
        required=True,
        help='Path to save the preprocessed masks',
    )
    args = parser.parse_args()
    cfp_path = Path(args.cfp_path)
    mask_path = Path(args.mask_path)
    out_pre_path = Path(args.out_pre_path)
    out_mask_path = Path(args.out_mask_path)
    for cfp_fn in sorted(cfp_path.iterdir()):
        print(cfp_fn)
        mask_fn = mask_path / cfp_fn.name
        img_enh_clahe, mask_enh = preprocess_img(cfp_fn, mask_fn)
        img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
        mask_enh = (mask_enh > 127).astype('uint8') * 255
        out_pre_fn = out_pre_path / cfp_fn.name
        out_pre_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(out_pre_fn, img_enh_clahe)
        out_mask_fn = out_mask_path / cfp_fn.name
        out_mask_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(out_mask_fn, mask_enh)

