# Data preparation
To extract images data and meta information in ProTECT/UMICH dataset.            
CT scans stored in dicom format and annotations by clinicians will be extracted and a matlab structure array will be used to store annotated patient's head CT. Fields in the structure array includes: 'brains' (brain images after standard contrast adjustment);  'annots': manual hematoma annotations from clinicians; 'dicomImgs': images in CT value; 'meta': meta data such as pixel spacing and slice thickness. 'Pid', patient id, 'Datatype': the dataset name.     


## How to use?
1. Generate struct arrays for annotated cases   
run preprocess_annotated_images.mlx      
2. Generate struct arrays for unannotated cases   
run runPreprocess.m      

## Preprocessing Steps (`preprocess_annotated_images.mlx`)
(Before running script) organize annotated data so that each patient has it's own folder with 
the patient id as the folder name. Within each patient's folder should be 
a subfolder named `DICOM` which contains the individual slices in `DICOM` 
format. The annotated images are not in a subfolder. Currently, only 
annotations in `.tiff` or `.tif` format are allowed.
