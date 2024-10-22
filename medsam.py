import argparse
import warnings
from pathlib import Path
from pprint import pprint
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from operator import itemgetter

from segment_anything import sam_model_registry
from dataset import SliceDataset  

from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import DiceLoss, CrossEntropy, CEDiceLoss, FocalDiceLoss, FocalLoss
from shutil import copytree, rmtree

def get_bounding_boxes(gt: Tensor) -> list[tuple[int, int, int, int]]:
    bounding_boxes = []
    for batch in gt:
        for class_idx in range(1, batch.shape[0]):  
            print(class_idx)
            class_mask = batch[class_idx]
            non_zero_coords = torch.nonzero(class_mask)
            if len(non_zero_coords) > 0:
                x_min, y_min = torch.min(non_zero_coords, dim=0)[0]
                x_max, y_max = torch.max(non_zero_coords, dim=0)[0]
                bounding_boxes.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
            else:
                bounding_boxes.append((0, 0, 0, 0)) 
    return bounding_boxes



def initialize_new_layers(medsam_model):
    nn.init.xavier_uniform_(medsam_model.mask_decoder.mask_tokens.weight)

    # Initialize missing layers for the output hypernetworks (indices 4 and 5)
    for i in range(4, 6):
        for layer in medsam_model.mask_decoder.output_hypernetworks_mlps[i].layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # Initialize IOU prediction head layer (layer 2)
    nn.init.xavier_uniform_(medsam_model.mask_decoder.iou_prediction_head.layers[2].weight)
    nn.init.zeros_(medsam_model.mask_decoder.iou_prediction_head.layers[2].bias)

def setup(args):
    gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")
    K: int = args.classes
    MedSAM_CKPT_PATH = "model/medsam_vit_b.pth"
    
    
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    initialize_new_layers(medsam_model)
    medsam_model = medsam_model.to(device)
    
    
    optimizer = torch.optim.Adam(medsam_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    
    # Dataset part
    B: int = args.batch_size
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),  
        lambda img: np.array(img)[np.newaxis, ...],  
        lambda nd: nd / 255,  
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: nd / (255 / (K- 1)) if K != 5 else nd / 63,
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...], 
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

    return (medsam_model, optimizer, scheduler, device, train_loader, val_loader, K)

def runTraining(args):
    print(f">>> Setting up to train MedSAM on {args.dataset}")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)

    if args.loss == "CE":
        loss_fn = CrossEntropy(idk=list(range(K)))
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
    log_hausdorff_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_hausdorff_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    
    # best_dice: float = 0
    best_metric: float = float('inf') if args.opt_metric == 'hausdorff' else 0.0
    
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
                    log_hausdorff = log_hausdorff_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                    log_hausdorff = log_hausdorff_val
                    
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    bounding_boxes = get_bounding_boxes(gt)

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape
                    
                    original_sizes = [image.shape[1:] for image in img]  
                    # batched_input = [{"image": image, "original_size": original_size} for image, original_size in zip(img, original_sizes)]
                    batched_input = [{"image": image, "original_size": original_size, "boxes": bbox}
                                     for image, original_size, bbox in zip(img, original_sizes, bounding_boxes)]
                    pred_logits = net(batched_input, multimask_output=True)
                    print("finish")
                    low_res_logits = torch.stack([pred["low_res_logits"] for pred in pred_logits]) 
                    
                    low_res_logits = low_res_logits.squeeze(1)                  
                    pred_probs = F.softmax(1 * low_res_logits, dim=1)
                    
                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)
                    
                    if args.loss == "Focal" or args.loss == "FocalDice":
                        loss = loss_fn(pred_probs, gt, pred_seg)
                    else:
                        loss = loss_fn(pred_probs, gt)           
                 
                    log_loss[e, i] = loss.item()

                    if opt:  # Only for training
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        
                        
                    
                    # Validation
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
                    if args.opt_metric == 'hausdorff':
                        postfix_dict: dict[str, str] = {"Hausdorff": f"{log_hausdorff[e, :j, 1:].mean():05.3f}",
                                                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    else:
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
        # np.save(args.dest / "hausdorff_tra.npy", log_hausdorff_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        # np.save(args.dest / "hausdorff_val.npy", log_hausdorff_val)

        # current_dice: float = log_dice_val[e, :, 1:].mean().item()
        # if current_dice > best_dice:
        #     print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
        #     best_dice = current_dice
        #     with open(args.dest / "best_epoch.txt", 'w') as f:
        #             f.write(str(e))
        
        # Hausdorff
        if args.opt_metric == 'hausdorff':
            current_metric: float = log_hausdorff_val[e, :, :].mean().item()
            is_better = current_metric < best_metric  # For Hausdorff, minimize
        else:  # Dice
            current_metric: float = log_dice_val[e, :, 1:].mean().item()
            is_better = current_metric > best_metric  # For Dice, maximize

        if is_better:
            print(f">>> Improved {args.opt_metric} at epoch {e}: {best_metric:05.3f}->{current_metric:05.3f}")
            best_metric = current_metric
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

def runTest(args):
    print(f">>> Running test on {args.dataset}")
    model, _, _, device, _, test_loader = setup(args)

    model.load_state_dict(torch.load(args.test_model))
    model.eval()

    with torch.no_grad():
        log_dice = 0
        for i, data in enumerate(tqdm_(test_loader, desc="Testing")):
            images, gts = data['images'].to(device), data['gts'].to(device)
            outputs = model(images)
            pred_seg = probs2one_hot(outputs)

            log_dice += dice_coef(pred_seg, gts)

        print(f"Mean Dice Score: {log_dice / len(test_loader)}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=['TOY2', 'SEGTHOR', 'SEGTHOR_test'])
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    parser.add_argument('--opt_metric', default='dice', choices=['dice', 'hausdorff'],
                        help="Metric to optimize: 'dice' for Dice score or 'hausdorff' for Hausdorff distance.")
    parser.add_argument('--test', action='store_true',
                        help="Run the test phase after training on the SEGTHOR/test dataset.")
    parser.add_argument('--only_test', action='store_true', help="Run only the test phase without training.")
    parser.add_argument('--transfer_learning', default=False, action='store_true')
    parser.add_argument('--pretrained_weights', default=False, action='store_true')
    parser.add_argument('--loss', default='CE', choices=['CE', 'Focal', 'Dice', 'CEDice', 'FocalDice'])
    parser.add_argument('--model', default='ENet', choices=['ENet','VMUNet','SwinUnet'])
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--classes', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    
    args = parser.parse_args()
    
    if args.test:
        runTest(args)
    else:
        runTraining(args)
    

if __name__ == '__main__':
    main()
    
    
