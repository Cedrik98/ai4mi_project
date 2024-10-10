import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import glob

def load_nifti(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    
    if union == 0:
        return 1.0
    else:
        return (2.0 * intersection) / union

def process_patient(gt_file, pred_file, num_segments):
    print(f"Processing {gt_file} and {pred_file}")
    
    gt = load_nifti(gt_file)
    pred = load_nifti(pred_file)
    
    # Normalize prediction values (when using their slicing code)
    pred = (pred / 63).astype(int)
    
    # 3D Dice score 
    dice_3d = {}
    for label in range(1, num_segments):  # 0 is background
        gt_label = (gt == label).astype(int)
        pred_label = (pred == label).astype(int)
        dice_3d[label] = dice_score(gt_label, pred_label)
    
    # 2D Dice score for each slice 
    num_slices = gt.shape[0]
    dice_2d = {label: [] for label in range(1, num_segments)}
    overall_dice_2d = []
    
    for slice_idx in range(num_slices):
        gt_slice = gt[slice_idx]
        pred_slice = pred[slice_idx]
        
        # Calculate 2D Dice for each segment
        for label in range(1, num_segments):
            gt_label_slice = (gt_slice == label).astype(int)
            pred_label_slice = (pred_slice == label).astype(int)
            dice_2d[label].append(dice_score(gt_label_slice, pred_label_slice))
        
        # Calculate overall 2D Dice 
        gt_all_slice = (gt_slice > 0).astype(int)
        pred_all_slice = (pred_slice > 0).astype(int)
        overall_dice_2d.append(dice_score(gt_all_slice, pred_all_slice))
    
    # Calculate (all segments combined)
    gt_all = (gt > 0).astype(int)
    pred_all = (pred > 0).astype(int)
    overall_dice_3d = dice_score(gt_all, pred_all)
    
    print("3D Dice Scores (whole volume):")
    for label, score in dice_3d.items():
        print(f"  Segment {label}: {score:.4f}")
    print()
    
    print("2D Dice Scores (per slice):")
    for label in range(1, num_segments):
        avg_dice = np.mean(dice_2d[label])
        std_dice = np.std(dice_2d[label])
        print(f"  Segment {label}: Mean = {avg_dice:.4f}, Std = {std_dice:.4f}")
    
    print("\nOverall Dice Scores:")
    print(f"  3D (whole volume): {overall_dice_3d:.4f}")
    print(f"  2D (per slice): Mean = {np.mean(overall_dice_2d):.4f}, Std = {np.std(overall_dice_2d):.4f}")
    
    return overall_dice_3d, overall_dice_2d

def compute_overall_dice_3d(dice_3d_scores):
    """Compute the mean and std of the overall 3D Dice score across all patients."""
    dice_array = np.array(dice_3d_scores)
    mean_dice = np.mean(dice_array)
    std_dice = np.std(dice_array)
    print(f"\n\nOverall 3D Dice Score across all patients:")
    print(f"  Mean = {mean_dice:.4f}, Std = {std_dice:.4f}")
    return mean_dice, std_dice

def compute_overall_dice_2d(dice_2d_scores):
    """Compute the mean and std of the 2D Dice score across all patients."""
    flattened_scores = [score for patient_scores in dice_2d_scores for score in patient_scores]
    dice_array = np.array(flattened_scores)
    mean_dice = np.mean(dice_array)
    std_dice = np.std(dice_array)
    print(f"Overall 2D Dice Score across all patients:")
    print(f"  Mean = {mean_dice:.4f}, Std = {std_dice:.4f}")
    return mean_dice, std_dice

def main(args: argparse.Namespace):
    pred_files = glob.glob(f"{args.pred_dir}/Patient_*.nii.gz")
    
    overall_dice_3d_list = []
    overall_dice_2d_list = []

    for pred_file in pred_files:
        patient_filename = Path(pred_file).name
        
        patient_id = patient_filename.replace(".nii.gz", "")
        
        gt_file = Path(args.gt_dir) / patient_id / "GT_fixed.nii.gz"
        
        if gt_file.exists():
            dice_3d, dice_2d = process_patient(gt_file, pred_file, args.num_segments)
            overall_dice_3d_list.append(dice_3d)
            overall_dice_2d_list.append(dice_2d)
        else:
            print(f"Warning: GT file {gt_file} not found, skipping.")
    
    compute_overall_dice_3d(overall_dice_3d_list)
    compute_overall_dice_2d(overall_dice_2d_list)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dice calculation parameters')
    parser.add_argument('--gt_dir', type=str, required=True, help="Directory containing GT files")
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing predicted files")
    parser.add_argument('--num_segments', type=int, default=5, help="Number of segments (including background)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(get_args())
