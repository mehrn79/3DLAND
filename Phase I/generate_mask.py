import os
import nibabel as nib
import numpy as np
import cv2
import glob
from natsort import natsorted
from organList import *  # لیست اندام‌ها مانند Organ = ['liver', 'spleen', ...]

# اندام‌های مورد نظر برای استخراج ماسک
target_organs = ["liver", "spleen", "kidney_right", "kidney_left", "gallbladder", "stomach", "pancreas"]

# مسیرها
NIFTI_data_dir = '/media/external_10T/mehran_advand/segment/myenv/monai_wholeBody_ct_segmentation/Segmentation_Output'
DCM_data_dir = '/media/external_10T/mehran_advand/DeepLesion/Images_dicom_test'
output_dir = 'MONAI/'

# دریافت لیست بیماران (فولدرهایی مثل 000001_01_01)
patient_folders = natsorted(os.listdir(NIFTI_data_dir))

for patient_id in patient_folders:

    # if patient_id != '000053_06_01':
    #      continue

    print(f"\n🧾 Processing patient: {patient_id}")
    
    # ساخت مسیر فایل NIfTI و فولدر DICOM
    nii_path = os.path.join(NIFTI_data_dir, patient_id, f"{patient_id}_trans.nii.gz")
    dcm_folder = os.path.join(DCM_data_dir, patient_id)
    
    if not os.path.isfile(nii_path):
        print(f"❌ NIfTI file not found for {patient_id}")
        continue

    dcm_files = glob.glob(os.path.join(dcm_folder, "*.dcm"))
    if not dcm_files:
        print(f"❌ No DICOM files found for {patient_id}")
        continue
    
    dcm_files = natsorted(dcm_files)
    
    # بارگذاری داده‌ی NIfTI
    print(f"📥 Reading NIfTI: {nii_path}")
    nii = nib.load(nii_path)
    label_data = nii.get_fdata()

    # بررسی و تصحیح محورها بر اساس affine
    if nii.affine[0, 0] > 0:
        label_data = np.flip(label_data, axis=0)
    if nii.affine[1, 1] > 0:
        label_data = np.flip(label_data, axis=1)
    if nii.affine[2, 2] > 0:
        label_data = np.flip(label_data, axis=2)

    label_data = np.transpose(label_data, (2, 1, 0))  # Z, Y, X
    num_slices = min(len(label_data), len(dcm_files))

    # پردازش هر اندام
    for target_organ in target_organs:
        if target_organ not in Organ:
            print(f"⚠️ Organ '{target_organ}' not in Organ list. Skipping...")
            continue

        organ_index = Organ.index(target_organ)
        print(f"🧠 Processing organ: {target_organ} (index {organ_index})")

        for idx in range(num_slices):
            binary_mask = (label_data[idx] == organ_index).astype(np.uint8) * 255
            #if np.any(binary_mask):
            dcm_path = dcm_files[idx]
            out_path = dcm_path.replace(DCM_data_dir, output_dir)
            out_path = out_path.replace(patient_id, f"{patient_id}/MONAI_{target_organ}")
            out_path = out_path.replace('.dcm', '_OUT.png')
            out_path = out_path.replace('\\', '/')

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, binary_mask)