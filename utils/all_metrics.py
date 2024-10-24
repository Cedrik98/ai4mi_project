import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import json
from scipy.spatial.distance import directed_hausdorff

def load_nifti(file_path):
    return sitk.ReadImage(file_path)

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    return (2.0 * intersection) / union if union > 0 else 1.0

def dice_score_2d(gt, pred):
    dice_scores = []
    for i in range(gt.shape[0]):
        intersection = np.sum(gt[i] * pred[i])
        union = np.sum(gt[i]) + np.sum(pred[i])
        dice_scores.append((2.0 * intersection) / union if union > 0 else 1.0)
    return np.mean(dice_scores)

def voxel_to_mm(distance, spacing):
    return distance * np.sqrt(np.sum(np.array(spacing)**2))

def compute_hausdorff(gt, pred, spacing):
    labels_gt = np.unique(gt)
    labels_pred = np.unique(pred)
    
    hausdorff_distances = {}

    for label_gt, label_pred in zip(labels_gt[labels_gt > 0], labels_pred[labels_pred > 0]):
        mask_gt = (gt == label_gt).astype(int)
        mask_pred = (pred == label_pred).astype(int)
        
        coords_gt = np.column_stack(np.where(mask_gt))
        coords_pred = np.column_stack(np.where(mask_pred))

        if len(coords_gt) == 0 or len(coords_pred) == 0:
            hausdorff_distances[int(label_gt)] = 0
            continue

        # Calculate Hausdorff distance in voxel space
        hausdorff_distance_gt_to_pred = directed_hausdorff(coords_gt, coords_pred)[0]
        hausdorff_distance_pred_to_gt = directed_hausdorff(coords_pred, coords_gt)[0]
        hausdorff_distance_voxel = max(hausdorff_distance_gt_to_pred, hausdorff_distance_pred_to_gt)
        
        # Convert to millimeters 
        hausdorff_distance_mm = voxel_to_mm(hausdorff_distance_voxel, spacing)
        hausdorff_distances[int(label_gt)] = hausdorff_distance_mm

    return hausdorff_distances

def compute_volumetric(gt, pred, voxel_size):
    labels_gt = np.unique(gt[gt > 0])
    labels_pred = np.unique(pred[pred > 0])
    
    vs_results = {}
    
    for label_gt, label_pred in zip(labels_gt, labels_pred):
        volume_gt = np.sum(gt == label_gt) * voxel_size
        volume_pred = np.sum(pred == label_pred) * voxel_size
    
        if volume_gt + volume_pred > 0:
            vs = 1 - abs(volume_gt - volume_pred) / (volume_gt + volume_pred)
        else:
            vs = 0
    
        vs_results[int(label_gt)] = vs

    return vs_results

def compute_average_surface_distance(gt, pred, spacing):
    labels_gt = np.unique(gt)
    labels_pred = np.unique(pred)
    
    average_surface_distances = {}
    
    for label_gt, label_pred in zip(labels_gt[labels_gt > 0], labels_pred[labels_pred > 0]):
        seg_gt_mask = (gt == label_gt).astype(np.uint8)
        seg_pred_mask = (pred == label_pred).astype(np.uint8)
    
        seg_gt_mask_img = sitk.GetImageFromArray(seg_gt_mask)
        seg_pred_mask_img = sitk.GetImageFromArray(seg_pred_mask)
        
        # Set the spacing for both images
        seg_gt_mask_img.SetSpacing(spacing)
        seg_pred_mask_img.SetSpacing(spacing)
    
        seg_gt_surface = sitk.LabelContour(seg_gt_mask_img)
        seg_pred_surface = sitk.LabelContour(seg_pred_mask_img)
    
        surface_distance_filter = sitk.HausdorffDistanceImageFilter()
        surface_distance_filter.Execute(seg_gt_surface, seg_pred_surface)
        avg_distance = surface_distance_filter.GetAverageHausdorffDistance()
        average_surface_distances[int(label_gt)] = avg_distance  # This is already in mm

    return average_surface_distances

def false_positive_rate(gt, pred):
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def false_negative_rate(gt, pred):
    fn = np.sum((pred == 0) & (gt == 1))
    tp = np.sum((pred == 1) & (gt == 1))
    return fn / (fn + tp) if (fn + tp) > 0 else 0

def recall(gt, pred):
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_metrics(gt_file, pred_file):
    print(f"Loading images...")
    gt_img = load_nifti(gt_file)
    pred_img = load_nifti(pred_file)
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)
    voxel_size = np.prod(gt_img.GetSpacing())
    spacing = gt_img.GetSpacing()
    print(f"Images loaded.")
    
    metrics = {
        "dice_score": {},
        "dice_score_2d": {},
        "hausdorff_distance": {},
        "volumetric_similarity": {},
        "false_positive_rate": {},
        "false_negative_rate": {},
        "average_surface_distance": {},
        "recall": {}
    }
    
    print(f"Calculating overall metrics...")
    gt_all = (gt > 0).astype(int)
    pred_all = (pred > 0).astype(int)
    metrics["overall"] = {
        "dice_score": dice_score(gt_all, pred_all),
        "dice_score_2d": dice_score_2d(gt_all, pred_all),
        "false_positive_rate": false_positive_rate(gt_all, pred_all),
        "false_negative_rate": false_negative_rate(gt_all, pred_all),
        "recall": recall(gt_all, pred_all)
    }
    print(f"Overall metrics calculated.")
    
    print(f"Calculating Hausdorff distance...")
    hausdorff_distances = compute_hausdorff(gt, pred, spacing)
    metrics["hausdorff_distance"] = hausdorff_distances
    metrics["overall"]["hausdorff_distance"] = np.mean(list(hausdorff_distances.values()))
    
    print(f"Calculating volumetric similarity...")
    volumetric_similarities = compute_volumetric(gt, pred, voxel_size)
    metrics["volumetric_similarity"] = volumetric_similarities
    metrics["overall"]["volumetric_similarity"] = np.mean(list(volumetric_similarities.values()))
    
    print(f"Calculating average surface distance...")
    average_surface_distances = compute_average_surface_distance(gt, pred, spacing)
    metrics["average_surface_distance"] = average_surface_distances
    metrics["overall"]["average_surface_distance"] = np.mean(list(average_surface_distances.values()))
    
    labels_gt = np.unique(gt)
    labels_pred = np.unique(pred)
    
    # Create a mapping from actual labels to output labels
    label_mapping = {label: i+1 for i, label in enumerate(sorted(labels_gt[labels_gt > 0]))}
    
    for label_gt, label_pred in zip(labels_gt[labels_gt > 0], labels_pred[labels_pred > 0]):
        print(f"Processing label {label_gt}...")
        gt_label = (gt == label_gt).astype(int)
        pred_label = (pred == label_pred).astype(int)
        
        output_label = label_mapping[label_gt]
        metrics["dice_score"][output_label] = dice_score(gt_label, pred_label)
        metrics["dice_score_2d"][output_label] = dice_score_2d(gt_label, pred_label)
        metrics["false_positive_rate"][output_label] = false_positive_rate(gt_label, pred_label)
        metrics["false_negative_rate"][output_label] = false_negative_rate(gt_label, pred_label)
        metrics["recall"][output_label] = recall(gt_label, pred_label)
        
        print(f"Label {label_gt} processed.")
    
    # Update the other metric dictionaries to use the new label mapping
    metrics["hausdorff_distance"] = {label_mapping[k]: v for k, v in metrics["hausdorff_distance"].items()}
    metrics["volumetric_similarity"] = {label_mapping[k]: v for k, v in metrics["volumetric_similarity"].items()}
    metrics["average_surface_distance"] = {label_mapping[k]: v for k, v in metrics["average_surface_distance"].items()}
    
    print(f"All metrics calculated.")
    
    return metrics

def process_folder(gt_folder, pred_folder):
    gt_files = sorted(glob(os.path.join(gt_folder, "*.nii.gz")))
    pred_files = sorted(glob(os.path.join(pred_folder, "*.nii.gz")))
    
    if len(gt_files) != len(pred_files):
        raise ValueError("Number of ground truth and prediction files do not match")
    
    results = {"patients": {}, "average": {}}

    for gt_file, pred_file in zip(gt_files, pred_files):
        patient_id = os.path.basename(gt_file).split('.')[0]
        print(f"Processing: {patient_id}")
        results["patients"][patient_id] = calculate_metrics(gt_file, pred_file)
    
    print("Calculating average results...")
    results["average"]["overall"] = {}
    for metric in results["patients"][list(results["patients"].keys())[0]]["overall"].keys():
        results["average"]["overall"][metric] = np.mean([patient["overall"][metric] for patient in results["patients"].values()])
    
    for metric in results["patients"][list(results["patients"].keys())[0]].keys():
        if metric != "overall":
            results["average"][metric] = {}
            all_labels = set()
            for patient in results["patients"].values():
                all_labels.update(patient[metric].keys())
            for label in all_labels:
                values = [patient[metric][label] for patient in results["patients"].values() if label in patient[metric]]
                results["average"][metric][label] = np.mean(values) if values else None
    
    return results

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_results_to_json(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    serializable_results = convert_to_serializable(results)
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

if __name__ == "__main__":
    # Created test set of 5 patients as Nifti files
    gt_folder = "internal_test_set/gt"
    # Change {model} with name of folder of trained model
    pred_folder = "results/segthor/{model}/predictions"
    output_file = "results/segthor/all_metrics_jsons/{model}.json"
    
    results = process_folder(gt_folder, pred_folder)
    save_results_to_json(results, output_file)
    print(f"Results saved to {output_file}")