from typing import Any
from pathlib import Path
import argparse
from argparse import Namespace
import json
import gc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import functional as F
from torch import Tensor
from torch import nn
from torchvision import utils as vutils
from skimage import io
from skimage.transform import resize

from transformations import to_torch_tensors, pad_images_unet
from model import RRWNet
import preprocessing



########################################################################
# Dataclasses


@dataclass
class DataPaths:
    cfp: Path
    pre: Path | None
    mask: Path | None


@dataclass
class ResizedImage:
    img: NDArray
    ohw: tuple[int, int] | None  # original height and width


########################################################################
# Functions


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--cfp_path',
        type=str, required=True,
        help='Path to input CFP images (required)'
    )
    parser.add_argument(
        '-t', '--model_type',
        type=str, choices=['av', 'bv'], default='av',
        help='Type of model to use: av (artery/vein) or bv (blood vessels) [default: %(default)s]'
    )
    parser.add_argument(
        '-w', '--weights_path',
        type=str, default='./__weights',
        help='Path to weights directory [default: %(default)s]'
    )
    parser.add_argument(
        '-s', '--save_path',
        type=str, default='./__predictions',
        help='Path to save predictions [default: %(default)s]'
    )
    parser.add_argument(
        '-m', '--masks_path',
        type=str, default=None,
        help='Path to the masks [default: %(default)s]'
    )
    parser.add_argument(
        '-p', '--pre_path',
        type=str, default=None,
        help='Path to the preprocessed images [default: %(default)s]'
    )
    parser.add_argument(
        '-g', '--use-gave-format',
        action='store_true', default=False,
        help='Save output in GAVE format (R for arteries, B for all'
            ' blood vessels, G for veins); otherwise the format is'
            ' as in RRWNet (R for arteries, G for veins, B for all'
            ' blood vessels) [default: %(default)s]'
    )
    parser.add_argument(
        '--tta',
        action='store_true', default=False,
        help='Use test time augmentation [default: %(default)s]'
    )
    return parser.parse_args()


def get_models_prediction(model: nn.Module, tensor: Tensor) -> Tensor:
    pred = model(tensor)
    if isinstance(pred, list):
        pred = pred[-1]
    else:
        pred = pred
    return pred


def get_predictions(
    model: nn.Module,
    tensor: Tensor,
    mask_tensor: Tensor,
    model_type: str,
    test_aug: bool = True,
) -> Tensor:
    original_tensor = tensor.clone()
    if test_aug:
        print('  Using test time augmentation: rotations and flips')
        tensors = []
        # Rotate 90, 180, 270 degrees, and flip horizontally and vertically
        for angle in [0, 90, 180, 270]:
            rotated_tensor = torch.rot90(original_tensor, k=angle // 90, dims=(2, 3))
            for flip in [False, True]:
                if flip:
                    flipped_tensor = torch.flip(rotated_tensor, dims=(3,))
                else:
                    flipped_tensor = rotated_tensor.clone()
                tensors.append((flipped_tensor, angle, flip))
    else:
        tensors = [(tensor, 0, False)]
    all_preds = []
    for tensor, angle, flip in tensors:
        pred = get_models_prediction(model, tensor)
        pred = torch.sigmoid(pred)
        if flip:
            pred = torch.flip(pred, dims=(3,))
        if angle > 0:
            pred = torch.rot90(pred, k=-angle // 90, dims=(2, 3))
        pred[mask_tensor < 0.5] = 0
        all_preds.append(pred)
    # Combine predictions from all augmentations by majority voting
    if len(all_preds) > 1:
        print(f'  Combining {len(all_preds)} predictions')
        a = torch.stack([x[:, 0] for x in all_preds], dim=1)
        v = torch.stack([x[:, 1] for x in all_preds], dim=1)
        bv = torch.stack([x[:, 2] for x in all_preds], dim=1)
        if model_type == 'bv':
            print('  Using bv specific combination')
            a_uni = torch.mean(a, dim=1, keepdim=True)
            v_uni = torch.mean(v, dim=1, keepdim=True)
            av = (a + v).clamp(min=0, max=1)
            bv_av = torch.concatenate([bv, av], dim=1)
            # print(f'  bv_av shape: {bv_av.shape}')
            bv_uni = torch.mean(bv_av, dim=1, keepdim=True)
            a_uni[a_uni > 0.5] = a.max(dim=1, keepdim=True).values[a_uni > 0.5]
            # a_uni[a_uni <= 0.5] = a.min(dim=1, keepdim=True).values[a_uni <= 0.5]
            v_uni[v_uni > 0.5] = v.max(dim=1, keepdim=True).values[v_uni > 0.5]
            # v_uni[v_uni <= 0.5] = v.min(dim=1, keepdim=True).values[v_uni <= 0.5]
            bv_uni[bv_uni > 0.5] = bv_av.max(dim=1, keepdim=True).values[bv_uni > 0.5]
            # bv_uni[bv_uni <= 0.5] = bv.min(dim=1, keepdim=True).values[bv_uni <= 0.5]
        # elif option == 1:
        #     a_uni = torch.median(a, dim=1, keepdim=True).values
        #     v_uni = torch.median(v, dim=1, keepdim=True).values
        #     bv_uni = torch.median(bv, dim=1, keepdim=True).values
        else:
            print('  Using simple mean combination')
            a_uni = torch.mean(a, dim=1, keepdim=True)
            v_uni = torch.mean(v, dim=1, keepdim=True)
            bv_uni = torch.mean(bv, dim=1, keepdim=True)
        pred_uni = torch.cat([a_uni, v_uni, bv_uni], dim=1)
        # Assign the maximum value to the artery channel, if the mean
        # value is greater than 0.5, otherwise assign 1 - value
        # a_uni = torch.where(a_uni > 0.5, a.max(dim=0, keepdim=True)[0], 1 - a.min(dim=0, keepdim=True)[0])
    else:
        # option = 1
        # if option == 0:
        #     a_uni = all_preds[0][:, 0:1]
        #     v_uni = all_preds[0][:, 1:2]
        #     bv_uni = all_preds[0][:, 2:3]
        #     # print(f' {a_uni.shape}, {v_uni.shape}, {bv_uni.shape}')
        #     bv_uni = torch.max(torch.stack([bv_uni, a_uni, v_uni], dim=1), dim=1).values
        #     pred_uni = torch.cat([a_uni, v_uni, bv_uni], dim=1)
        # else:
        pred_uni = all_preds[0]
    return pred_uni


def in_parent_or_none(ref_path: Path, this_path: str | None, default: str) -> Path | None:
    if this_path is None:
        path = ref_path.parent / default
    else:
        if Path(this_path).exists():
            path = Path(this_path)
        elif (ref_path.parent / this_path).exists():
            path = ref_path.parent / this_path
        else:
            path = None
    return path


def get_paths(args: Namespace) -> DataPaths:
    cfp_path = Path(args.cfp_path)
    assert cfp_path.exists(), cfp_path
    masks_path = in_parent_or_none(cfp_path, args.masks_path, 'mask')
    pre_path = in_parent_or_none(cfp_path, args.pre_path, 'pre')
    return DataPaths(
        cfp=cfp_path,
        pre=pre_path,
        mask=masks_path,
    )



def get_model(config: Namespace, checkpoint: dict, device) -> nn.Module:
    print('Creating model')
    model = RRWNet(
        config.in_channels,
        config.out_channels,
        config.base_channels,
        config.num_iterations
    )

    print('Loading weights')
    model.load_state_dict(checkpoint)

    model.eval()

    model.to(device)

    return model


def read_to_zero_one(img_fn) -> ResizedImage:
    img = io.imread(img_fn).astype(np.float32)
    if img.max() > 255:
        img = img / 65535.0
    if img.max() > 1:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    if len(img.shape) == 3:
        img = img[..., :3]
    h, w = img.shape[0], img.shape[1]
    original_spatial_shape = (h, w)
    if w != 1408:
        new_w = 1408
        new_h = int(h * (new_w / w))
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    else:
        original_spatial_shape = None
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    return ResizedImage(img, original_spatial_shape)


def get_corr_fn(ref_fn: Path, path: Path | None) -> Path | None:
    corr_fn = None
    if path is not None and path.exists():
        for corr_fn in path.iterdir():
            if corr_fn.stem == ref_fn.stem:
                break
    return corr_fn



########################################################################
# Functions


def main():
    args = get_args()
    use_cfp = args.model_type == 'bv'
    test_aug = args.tta

    save_path = Path(args.save_path) / args.model_type
    save_path.mkdir(exist_ok=True, parents=True)

    paths = get_paths(args)
    print('Paths:', paths)

    print(f'Loading model: {args.model_type}')
    checkpoint = torch.load(f'__weights/{args.model_type}.pth')

    print('Loading config')
    with open(f'__weights/{args.model_type}_config.json', 'r') as f:
        config = json.load(f)

    print('Config:')
    print(json.dumps(config, indent=4))

    # Namespace from config dict
    config = argparse.Namespace(**config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(config, checkpoint, device)

    for cfp_fn in sorted(paths.cfp.iterdir()):
        if not cfp_fn.is_file():
            continue
        mask_fn = get_corr_fn(cfp_fn, paths.mask)
        pre_fn = get_corr_fn(cfp_fn, paths.pre)
        print(f'> Processing {cfp_fn.name}')
        cfp_res = read_to_zero_one(cfp_fn)
        cfp = cfp_res.img
        print(f'  CFP shape: {cfp.shape}')
        if mask_fn is None:
            print(f'  Warning: No mask found, using thresholding')
            mask = (cfp.sum(axis=2) > 0.01).astype(np.float32)
        else:
            mask = read_to_zero_one(mask_fn).img.astype(np.float32)
        if pre_fn is not None:
            pre = read_to_zero_one(pre_fn).img
        else:
            print(f'  Warning: No preprocessed image found, preprocessing on the fly')
            pre, mask = preprocessing.preprocess_img(cfp, mask)
        print(f'  Preprocessed shape: {pre.shape}')
        if use_cfp:
            img = np.concatenate([pre, cfp], axis=-1)
        else:
            img = pre
        images, paddings = pad_images_unet([img, mask], return_paddings=True)
        img = images[0]
        mask = images[1]
        padding = paddings[0]
        # padding format: ((top, bottom), (left, right), (0, 0))
        with torch.no_grad():
            tensors = to_torch_tensors([img, mask], device=device)
            tensor = tensors[0]
            mask_tensor = tensors[1]
            pred = get_predictions(
                model,
                tensor,
                mask_tensor,
                model_type=args.model_type,
                test_aug=test_aug,
            )
            pred = pred[..., padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            if args.use_gave_format:
                pred_form = torch.zeros_like(pred)
                pred_form[:, 0] = pred[:, 0]
                pred_form[:, 1] = pred[:, 2]
                pred_form[:, 2] = pred[:, 1]
            else:
                pred_form = pred
            if cfp_res.ohw is not None:
                pred_form = F.interpolate(
                    pred_form,
                    size=cfp_res.ohw,
                    mode='bilinear',
                    align_corners=False
                )
            save_fn = save_path / Path(cfp_fn).name
            vutils.save_image(pred_form, save_fn)
            gc.collect()

    print('Images saved in', save_path)



if __name__ == '__main__':
    main()
