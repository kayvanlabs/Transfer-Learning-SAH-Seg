# Transfer-Learning-SAH-Seg
Evaluating different fine-tuning strategies for aneurysmal subarachnoid hematoma segmentation via a multi-view CNN and transfer learning.

## Steps
1. Organize your annotated axial head CT DICOM images and annotations so they are in folders named with the patient ID. Inside each patient ID folder, there should be all the slices of the CT as `.tif` files. These include the annotations. Annotations were completed in [MicroDicom](https://www.microdicom.com/) in the original work. Annotations should appear in yellow. A folder named 'DICOM' should contain the `.dcm` files for every slice in the corresponding series.
2. Run the `preprocess_annotated_images.mlx` script to preprocess the DICOM images and annotations
3. Run `sbatch job_tune_cv.py` with the desired arguments.

