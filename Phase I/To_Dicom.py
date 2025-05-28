import pyplastimatch as pypla
from pyplastimatch.utils.install import install_precompiled_binaries

install_precompiled_binaries("")

# convert one of the NIFTI images to DICOM: name: <patient1>, output folder: <dicom_output>
convert_args_ct = {
    "input": "datasets/004428_01_02_182-242.nii.gz",
    "patient-id": "patient1",
    "output-dicom": "dicom_output",
}
pypla.convert(verbose=True, **convert_args_ct)
