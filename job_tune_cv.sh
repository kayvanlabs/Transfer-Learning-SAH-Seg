#!/bin/bash
# The interpreter used to execute the script

# directives that convey submission options:

#SBATCH --job-name=tune_cv_hematoma_segmentation
#SBATCH --mail-user=hodgman@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --account=kayvan99
#SBATCH -p gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1

module load python/3.10.4
module load pytorch
module load matplotlib
pip install -U tensorly-torch
pip install -U tensorly

# The application(s) to execute along with its input arguments and options:
python tune_cv.py --name $1 --tune $2 --epoch_number 20 --trained_model_path $3