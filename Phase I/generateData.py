import numpy as np
import nibabel as nib
import os
import cv2
import csv
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset
import numpy as np
import os
import datetime
import zipfile

def slices2nifti(ims, fn_out, spacing):  
    """Save 2D slices to 3D NIfTI file considering the spacing."""  
    if len(ims) < 300:  # cv2.merge does not support too many channels  
        V = cv2.merge(ims)  
    else:  
        V = np.empty((ims[0].shape[0], ims[0].shape[1], len(ims)))  
        for i in range(len(ims)):  
            V[:, :, i] = ims[i]  

    # The transformation matrix suitable for 3D slicer and ITK-SNAP  
    T = np.array([[0, -spacing[1], 0, 0],   
                  [-spacing[0], 0, 0, 0],   
                  [0, 0, -spacing[2], 0],   
                  [0, 0, 0, 1]])  
    img = nib.Nifti1Image(V, T)  
    path_out = os.path.join(dir_out, fn_out)  
    nib.save(img, path_out)  
    return path_out 



def load_slices(dir, slice_idxs):
    """Load slices from 16-bit PNG files and return images with their filenames."""
    slice_idxs = np.array(slice_idxs)
    if not np.all(slice_idxs[1:] - slice_idxs[:-1] == 1):
        print(f"⚠️ Slice indices are not consecutive")


    ims = []
    filenames = []

    for slice_idx in slice_idxs:
        fn = f'{slice_idx:03d}.png'
        path = os.path.join(dir_in, dir, fn)
        im = cv2.imread(path, -1)  # -1 to preserve 16-bit depth
        assert im is not None, f'Error reading {path}'

        im_corrected = (im.astype(np.int32) - 32768).astype(np.int16)
        ims.append(im_corrected)
        filenames.append(fn.split('.')[0])

    return ims, filenames
 

def read_DL_info():  
    """Read spacings and image indices in DeepLesion."""  
    spacings = []  
    idxs = []  
    with open(info_fn, 'r') as csvfile:  # Use 'r' mode for reading text files  
        reader = csv.reader(csvfile)  
        rownum = 0  
        for row in reader:  
            if rownum == 0:  
                header = row  
                rownum += 1  
            else:  
                idxs.append([int(d) for d in row[1:4]])  
                spacings.append([float(d) for d in row[12].split(',')])  

    idxs = np.array(idxs)  
    spacings = np.array(spacings)  
    return idxs, spacings  

def nii_to_dicom(nii_path, output_folder, filenames):
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    affine = nii_img.affine
    num_slices = data.shape[2]

    assert len(filenames) == num_slices, "Length of filename list must match number of slices."

    nii_base = os.path.splitext(os.path.basename(nii_path))[0]
    nii_base = nii_base.split('.')[0]
    dicom_subfolder = os.path.join(output_folder, nii_base)
    os.makedirs(dicom_subfolder, exist_ok=True)

    for i in range(num_slices):
        filename = os.path.join(dicom_subfolder, filenames[i])+'.dcm'  # Use provided filename

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.0"
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        dt = datetime.datetime.now()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = "MR"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.StudyTime = dt.strftime('%H%M%S')

        ds.Rows, ds.Columns = data.shape[:2]
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [float(affine[0,3]), float(affine[1,3]), float(affine[2,3] + i)]
        ds.ImageOrientationPatient = [1,0,0,0,1,0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1

        pixel_array = data[:, :, i].astype(np.uint16)
        ds.PixelData = pixel_array.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(filename)


def extract_and_collect_main_folders(zip_root_dir):
    extracted_paths = []

    for zip_name in sorted(os.listdir(zip_root_dir)):
        zip_path = os.path.join(zip_root_dir, zip_name)

        # فقط فایل‌های .zip واقعی
        if zip_name.lower().endswith('.zip') and os.path.isfile(zip_path):
            try:
                extract_folder = os.path.join(zip_root_dir, zip_name.replace('.zip', ''))
                os.makedirs(extract_folder, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

                os.remove(zip_path)

                # حذف پوشه‌های مخفی
                extracted_subfolders = [
                    os.path.join(extract_folder, name)
                    for name in os.listdir(extract_folder)
                    if os.path.isdir(os.path.join(extract_folder, name)) and not name.startswith('.')
                ]

                if not extracted_subfolders:
                    extracted_paths.append(extract_folder)
                else:
                    extracted_paths.extend(extracted_subfolders)

            except zipfile.BadZipFile:
                print(f"⚠️ Skipping bad zip file: {zip_name}")
                continue

    return extracted_paths


# Main
zip_root_dir ='' 
folders = extract_and_collect_main_folders(zip_root_dir)
def find_image_folders(directory):  
    
    image_folders = []  
    
    # 
    for root, dirs, files in os.walk(directory):  
        for dir_name in dirs:  
            if dir_name.startswith("Images_png_"):  
                
                image_folders.append(os.path.join(root, dir_name))  
    
    return image_folders  

 
folder_path = ''   
result = find_image_folders(folder_path)  

print(result)

dir_out = ''
out_fmt = '%s.nii.gz'  # format of the nifti file name to output
info_fn =''  # file name of the information file
idxs, spacings = read_DL_info()  

for folder in result :
    dir_in = folder + '/Images_png'
    if not os.path.exists(dir_out):  
        os.mkdir(dir_out)  
    img_dirs = os.listdir(dir_in)  
    img_dirs.sort() 

    for dir1 in img_dirs:  
        #Find the image info according to the folder's name    

        idxs1 = np.array([int(d) for d in dir1.split('_')])  
        i1 = np.where(np.all(idxs == idxs1, axis=1))[0]  
        spacings1 = spacings[i1[0]]  

        fns = os.listdir(os.path.join(dir_in, dir1))  
        slices = [int(d[:-4]) for d in fns if d.endswith('.png')]  
        slices.sort()  

        groups = [slices] 

        for group in groups:  
            # Group contains slices indices of a sub-volume  
            ims,names = load_slices(dir1, group)  
            fn_out = out_fmt % (dir1)  
            path_out = slices2nifti(ims, fn_out, spacings1) 
            nii_to_dicom(path_out, "",names)




