# Transfer-Learning-SAH-Seg
Evaluating different fine-tuning strategies for aneurysmal subarachnoid hematoma segmentation via a multi-view CNN and transfer learning.

## Steps
1. Organize your annotated axial head CT DICOM images and annotations so they are in folders named with the patient ID. Inside each patient ID folder, there should be all the slices of the CT as `.tif` files. These include the annotations. Annotations were completed in [MicroDicom](https://www.microdicom.com/) in the original work. Annotations should appear in yellow. A folder named 'DICOM' should contain the `.dcm` files for every slice in the corresponding series.
2. Run the `preprocess_annotated_images.mlx` script to preprocess the DICOM images and annotations
3. Run `sbatch job_tune_cv.sh` with the desired arguments.

```
usage: tune_cv.py [-h] [--tune TUNE] [--name NAME] [--epoch_number EPOCH_NUMBER] [--level LEVEL] [--decomposition DECOMPOSITION] [--trained_model_path TRAINED_MODEL_PATH]
                  [--train]

Train/tune a model

optional arguments:
  -h, --help            show this help message and exit
  --tune TUNE           parts of model to tune
  --name NAME           experiment name
  --epoch_number EPOCH_NUMBER
                        number of epochs to tune
  --level LEVEL         level of model
  --decomposition DECOMPOSITION
                        decomposition method
  --trained_model_path TRAINED_MODEL_PATH
                        path to pre-trained model
  --train               train the model
```
