import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, ScaleIntensityd, EnsureTyped,
    Activationsd, AsDiscreted, Invertd, SaveImaged
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer

input_dir = "" #NIFTI Image
output_dir = ""
model_path = "models/model.pt"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
data_dicts = [{"image": f} for f in image_files]

pre_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True),
    ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
    EnsureTyped(keys=["image"]),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=105,  
    init_filters=32,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dropout_prob=0.2
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dataset = Dataset(data=data_dicts, transform=pre_transforms)
dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.25)

post_transforms = Compose([
    Activationsd(keys="pred", softmax=True),
    AsDiscreted(keys="pred", argmax=True),
    Invertd(
        keys="pred",
        transform=pre_transforms,
        orig_keys="image",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True
    ),
    SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        output_dir=output_dir,
        output_postfix="seg",
        separate_folder=False,
        resample=True  
    )
])

with torch.no_grad():
    for batch_data in dataloader:
        batch_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}
        outputs = inferer(inputs=batch_data["image"], network=model)
        batch_data["pred"] = outputs
        batch_data = post_transforms(batch_data)
        print("✅ Done:", batch_data["image_meta_dict"]["filename_or_obj"][0])
