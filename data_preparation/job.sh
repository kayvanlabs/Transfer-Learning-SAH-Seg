#!/bin/bash
#SBATCH --mail-user=your_email@university.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=6:00:00
#SBATCH --account=slurm_account_name
#SBATCH --partition=your_partition

module load matlab/R2022b
matlab -nodisplay -nodesktop -r 'runPreprocess'
