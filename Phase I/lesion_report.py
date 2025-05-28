import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

SEG_ROOT = "MONAI"
DL_INFO_PATH = ""
OUTPUT_PATH = ""
DEBUG_DIR = ""
os.makedirs(DEBUG_DIR, exist_ok=True)

TARGET_ORGANS = ["liver", "spleen", "kidney_right", "kidney_left", "gallbladder", "stomach", "pancreas"]
OVERLAP_THRESHOLD = 0.1

df = pd.read_csv(DL_INFO_PATH)
results = []


for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        filename = row["File_name"]
        patient_id, study_id, series_id = filename.split("_")[:3]
        series_name = f"{patient_id}_{study_id}_{series_id}"
        series_path = os.path.join(SEG_ROOT, series_name)

        if not os.path.exists(series_path):
            continue

        bbox = np.array(row["Bounding_boxes"].strip("[]").split(","), dtype=float).astype(int).tolist()
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        lesion_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
        if lesion_area == 0:
            continue

        slice_start, slice_end = map(int, str(row["Slice_range"]).strip().split(","))
        key_slice = int(row["Key_slice_index"])
        matched_organs = set()
        found_overlap = False

        overlap_slices = range(max(key_slice - 2, slice_start), min(key_slice + 2, slice_end) + 1)

        for organ in TARGET_ORGANS:
            monai_organ = f"MONAI_{organ}"

            for slice_idx in overlap_slices:
                slice_filename = f"{slice_idx:03}_OUT.png"
                mask_path = os.path.join(series_path, monai_organ, slice_filename)

                if not os.path.exists(mask_path):
                    continue

                mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_gray is None or mask_gray.shape[0] == 0:
                    continue

                mask_gray = cv2.rotate(mask_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask_gray = cv2.flip(mask_gray, 0)

                mask_crop = mask_gray[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
                overlap_area = np.sum(mask_crop > 0)
                overlap_ratio = overlap_area / lesion_area

                if overlap_ratio > OVERLAP_THRESHOLD:
                    matched_organs.add(organ)
                    found_overlap = True
                    break

            if not found_overlap:  
                proximity_distances = []  
                all_distances = []  

                for organ in TARGET_ORGANS:  
                    monai_organ = f"MONAI_{organ}"  
                    slice_filename = f"{key_slice:03}_OUT.png"  
                    mask_path = os.path.join(series_path, monai_organ, slice_filename)  

                    if not os.path.exists(mask_path):  
                        continue  

                    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
                    if mask_gray is None or np.sum(mask_gray > 0) == 0:  
                        continue  

                    mask_gray = cv2.rotate(mask_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)  
                    mask_gray = cv2.flip(mask_gray, 0)  

                    mask_bin = (mask_gray > 0).astype(np.uint8)  
                    dist_transform = cv2.distanceTransform(255 - mask_bin * 255, cv2.DIST_L2, 5)  

                    lesion_mask = np.zeros_like(mask_bin)  
                    lesion_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = 1  
                    lesion_dist = dist_transform * lesion_mask  

                    if np.any(lesion_dist > 0):  
                        min_dist = lesion_dist[lesion_dist > 0].min()  
                        all_distances.append((organ, min_dist))  
                        
                        if min_dist <= 10:
                            matched_organs.add(organ)
                            found_overlap = True
                            proximity_distances.clear()
                            

                        if min_dist >10 and min_dist <= 20 and not found_overlap  : 
                            proximity_distances.append((organ, min_dist))  


                for organ, _ in proximity_distances:  
                    matched_organs.add(organ + "_prox")  
                            
        results.append({
            "series": series_name,
            "slice_range": f"{slice_start}~{slice_end}",
            "key_slice": key_slice,
            "lesion_id": idx,
            "matched_organs": ";".join(sorted(matched_organs)) if matched_organs else "none"
        })

    except Exception as e:
        print(f"⚠️ error {idx}: {e}")
        continue


out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_PATH, index=False)

