import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
from scipy.ndimage import distance_transform_edt

def extract_border(mask):
    """Extract binary border from mask using morphological gradient."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    border = dilated - eroded
    return border

def surface_dice(gt_mask, pred_mask, tolerance=2):
    gt_border = extract_border((gt_mask > 0).astype(np.uint8))
    pred_border = extract_border((pred_mask > 0).astype(np.uint8))

    if not np.any(gt_border) or not np.any(pred_border):
        return np.nan

    gt_dist = distance_transform_edt(1 - gt_border)
    pred_dist = distance_transform_edt(1 - pred_border)

    gt_match = pred_dist[gt_border > 0] <= tolerance
    pred_match = gt_dist[pred_border > 0] <= tolerance

    tp = np.count_nonzero(gt_match) + np.count_nonzero(pred_match)
    total = np.count_nonzero(gt_border) + np.count_nonzero(pred_border)

    return tp / total if total > 0 else np.nan


def compute_metrics(gt_folder, pred_folder, output_csv="results.csv", tolerance=2):
    metrics = {
        "filename": [],
        "IoU": [],
        "Dice": [],
        "SurfaceDice": []
    }

    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".png")])

    for filename in tqdm(pred_files):
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(gt_folder, filename)

        print(pred_path,gt_path)

        if not os.path.exists(gt_path):
            print(f"⚠️ There is no {filename}")
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or pred_mask is None or gt_mask.shape != pred_mask.shape:
            print(f"⚠️ No file : {filename}")
            continue

        gt_bin = (gt_mask > 0).astype(np.uint8).flatten()
        pred_bin = (pred_mask > 0).astype(np.uint8).flatten()

        iou = jaccard_score(gt_bin, pred_bin, zero_division=0)
        dice = f1_score(gt_bin, pred_bin, zero_division=0)
        surf_dice = surface_dice(gt_mask, pred_mask, tolerance)

        metrics["filename"].append(filename)
        metrics["IoU"].append(iou)
        metrics["Dice"].append(dice)
        metrics["SurfaceDice"].append(surf_dice)

    df = pd.DataFrame(metrics)

    if len(df) > 0:
        mean_row = {
            "filename": "MEAN",
            "IoU": df["IoU"].mean(),
            "Dice": df["Dice"].mean(),
            "SurfaceDice": df["SurfaceDice"].mean()
        }
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    df.to_csv(output_csv, index=False)


compute_metrics(
    gt_folder="",
    pred_folder="",
    output_csv="",
    tolerance=2  
)
