#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset

from ENet import ENet
from model_utils.vmunet import VMUNet
from model_utils.vision_transformer import SwinUnet

import subprocess

from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import *

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = args.classes

    # ENet
    if args.model == 'ENet':
        net = ENet(1, K)
        net.init_weights()
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = None

    # SWIN
    if args.model == 'SwinUnet':
        if args.pretrained_weights:
            net = SwinUnet(img_size=224, num_classes=K)
            net.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
            if args.transfer_learning:
                net.setup_transfer_learning_swin()
        else:
            net = SwinUnet(img_size=256, num_classes=K)

        net.to(device)

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
        scheduler =  None
    
    # VMUNet
    if args.model == 'VMUNet':
        net = VMUNet()
        if args.pretrained_weights:
            net.load_from('./pretrained_ckpt/vmamba_tiny_e292.pth')
            if args.transfer_learning:
                net.setup_transfer_learning_vmunet()

        net.to(device)

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001, last_epoch=-1)


    # Dataset part
    B: int = args.batch_size
    root_dir = Path("data") / args.dataset

    if args.model == 'SwinUnet' and args.pretrained_weights:
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda img: np.array(img)[...],
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])
    else:
        img_transform = transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            lambda img: np.array(img)[...],
            # The idea is that the classes are mapped to {0, 255} for binary cases
            # {0, 85, 170, 255} for 4 classes
            # {0, 51, 102, 153, 204, 255} for 6 classes
            # Very sketchy but that works here and that simplifies visualization
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])


    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, scheduler, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)

    if args.loss == "CE":
        if args.mode == "full":
            loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
            loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)
    if args.loss == "Dice":
        loss_fn = DiceLoss(smooth=1) 
    if args.loss == "Focal":
        loss_fn = FocalLoss(alpha=0.25, gamma=2.5, idk=list(range(K)))
    if args.loss == "CEDice":
        loss_fn = CEDiceLoss(dice_weight=0.5, ce_weight=0.5, smooth=1, idk=list(range(K)))
    if args.loss == "FocalDice":
        loss_fn = FocalDiceLoss(dice_weight=0.5, focal_weight=0.5, smooth=1, alpha=0.25, gamma=2.5, idk=list(range(K)))

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

                    if args.loss == "Focal" or args.loss == "FocalDice":
                        loss = loss_fn(pred_probs, gt, pred_seg)
                    else:
                        loss = loss_fn(pred_probs, gt)           

                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                        for k in range(1, K)}

                    tq_iter.set_postfix(postfix_dict)

        if scheduler is not None:
            scheduler.step()

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

def runTest(args):
    """Run a segmentation test on the specified dataset using the best trained model. 
    Also creates segmentations from patients in test set"""
    
    print(f">>> Running test on {args.dataset} using the best model.")

    # Load the best model saved during training
    best_model_path = args.dest / "bestmodel.pkl"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Ensure training has been done and the best model is saved.")

    # Load the model
    net = torch.load(best_model_path)
    net.eval()

    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    net.to(device)

    K: int = args.classes
    B: int = args.batch_size

    root_dir = Path("data") / args.dataset

    # Prepare the test dataset and loader
    if args.model == 'SwinUnet' and args.pretrained_weights:
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda img: np.array(img)[...],
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])
    else:
        img_transform = transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            lambda img: np.array(img)[...],
            # The idea is that the classes are mapped to {0, 255} for binary cases
            # {0, 85, 170, 255} for 4 classes
            # {0, 51, 102, 153, 204, 255} for 6 classes
            # Very sketchy but that works here and that simplifies visualization
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])

    test_set = SliceDataset('test',
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)

    test_loader = DataLoader(test_set,
                             batch_size=B,
                             num_workers=args.num_workers,
                             shuffle=False)

    with torch.no_grad():
        tq_iter = tqdm_(enumerate(test_loader), total=len(test_loader), desc=">> Testing")
        for _, data in tq_iter:
            img = data['images'].to(device)

            pred_logits = net(img)
            pred_probs = F.softmax(1 * pred_logits, dim=1)
            
            # Save test predictions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                predicted_class: Tensor = probs2class(pred_probs)
                mult: int = 63 if K == 5 else (255 / (K - 1))
                save_images(predicted_class * mult,
                            data['stems'],
                            args.dest / "test_results")
    
    # Stitch segmented slices back into 3D volumes using stitch.py
    print(">>> Stitching slices back into 3D volumes...")
    stitch_command = [
        'python', 'stitch.py',
        '--data_folder', str(args.dest / "test_results"),
        '--dest_folder', str(args.dest / "stitched_volumes"),
        '--num_classes', "255",
        '--grp_regex', "(Patient_\\d\\d)_\\d\\d\\d\\d",
        '--source_scan_pattern', f"data/segthor_train/train/{{id_}}/GT_fixed.nii.gz"
    ]
    subprocess.run(stitch_command)
    print(f">> Saved stitched volumes to {args.dest}/test_results/stitched_volumes")

    # Calculate Dice score in 2D and 3D using dice_calculation.py
    print("\n>>> Calculating 2D and 3D Dice scores...")
    pred_folder = args.dest / "stitched_volumes"
    gt_folder = "data/segthor_train/train"

    dice_command = [
        'python', 'utils/dice_calculation.py',
        '--gt_dir', gt_folder,
        '--pred_dir', pred_folder,
        '--num_segments', str(K)
    ]
    subprocess.run(dice_command)

    print(f">>> Test completed. Results saved to {args.dest}/test_results.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=['TOY2', 'SEGTHOR'])
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    parser.add_argument('--only_test', action='store_true', help="Run only the test phase without training.")
    parser.add_argument('--transfer_learning', default=False, action='store_true')
    parser.add_argument('--pretrained_weights', default=False, action='store_true')
    parser.add_argument('--loss', default='CE', choices=['CE', 'Focal', 'Dice', 'CEDice', 'FocalDice'])
    parser.add_argument('--model', default='ENet', choices=['ENet', 'VMUNet', 'SwinUnet'])
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--classes', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    args = parser.parse_args()

    pprint(args)

    if args.only_test:
        runTest(args)
    else:
        runTraining(args)


if __name__ == '__main__':
    main()
