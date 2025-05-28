import os
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

CSV_PATH = ""
CT_IMAGES_ROOT = "" #deeplesion images
OUTPUT_MASK_DIR = "sam_lesion_2Dmasks_compound_box"
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type = "vit_h"
checkpoint_path = "sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)


df = pd.read_csv(CSV_PATH)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        series = "_".join(row["File_name"].split("_")[:3])
        key_slice = int(row["key_slice"])
        bbox = list(map(float, row["Bounding_boxes"].strip("[]").split(",")))
        bbox = list(map(int, bbox))
        x1, y1, x2, y2 = bbox

        slice_filename = f"{key_slice:03}.png"
        image_path = os.path.join(CT_IMAGES_ROOT, series, slice_filename)
        if not os.path.exists(image_path):
            print(f"There is no Image {image_path}")
            continue

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        outer_box = np.array([x1, y1, x2, y2])

        box_w, box_h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
       
        scale = 0.3  
        cw, ch = int(box_w * scale), int(box_h * scale)
        center_box = np.array([cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2])

        mask_outer, _, _ = predictor.predict(
            box=outer_box[None, :],
            multimask_output=False
        )

        mask_center, _, _ = predictor.predict(
            box=center_box[None, :],
            multimask_output=False
        )

        final_mask = np.logical_or(mask_outer[0], mask_center[0]).astype(np.uint8) * 255


        out_filename = f"{series}_{key_slice:03}.png"
        out_path = os.path.join(OUTPUT_MASK_DIR, out_filename)
        cv2.imwrite(out_path, final_mask)

    except Exception as e:
        print(f"⚠️ error{idx}: {e}")
