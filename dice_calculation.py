import numpy as np
import SimpleITK as sitk

def load_nifti(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    
    if union == 0:
        return 1.0
    else:
        return (2.0 * intersection) / union

def calculate_dice_metrics(gt_file, pred_file, num_segments):
    gt = load_nifti(gt_file)
    pred = load_nifti(pred_file)
    
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
    
    # Calculate oall segments combined)
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
    print()
    
    print("Overall Dice Scores:")
    print(f"  3D (whole volume): {overall_dice_3d:.4f}")
    print(f"  2D (per slice): Mean = {np.mean(overall_dice_2d):.4f}, Std = {np.std(overall_dice_2d):.4f}")

if __name__ == "__main__":
    gt_file = "Nnunet_data_correct/nnUNet_raw/Dataset700_Segthor/labelsTr/Patient_40.nii.gz"
    pred_file = "output_pp/Patient_40.nii.gz"
    num_segments = 5  # Total number of segments including background
    
    calculate_dice_metrics(gt_file, pred_file, num_segments)