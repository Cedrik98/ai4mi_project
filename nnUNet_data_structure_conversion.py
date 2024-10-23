import os
import shutil

# Paths
source_dir =  "train_correct" # currentnnu path
target_base_dir = "Nnunet_data_correct"  # nnUNet dir

# number ca
dataset_name = "Dataset700_Segthor"
task_dir = os.path.join(target_base_dir, "nnUNet_raw", dataset_name)

#  target directories
imagesTr_dir = os.path.join(task_dir, "imagesTr")
labelsTr_dir = os.path.join(task_dir, "labelsTr")

os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

# Loop through each patient folder
for patient_folder in sorted(os.listdir(source_dir)):
    patient_path = os.path.join(source_dir, patient_folder)

    if os.path.isdir(patient_path):
        # Define the file paths
        image_file = os.path.join(patient_path, f"{patient_folder}.nii.gz") 
        label_file = os.path.join(patient_path, "GT.nii.gz")  


        if os.path.exists(image_file) and os.path.exists(label_file):
            #  names in nnU-Net format
            new_image_name = f"{patient_folder}_0000.nii.gz" 
            new_label_name = f"{patient_folder}.nii.gz"

            # Copy to imagesTr
            shutil.copy(image_file, os.path.join(imagesTr_dir, new_image_name))

            # Copy  to labelsTr
            shutil.copy(label_file, os.path.join(labelsTr_dir, new_label_name))




