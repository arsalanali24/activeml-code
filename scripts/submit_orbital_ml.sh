#!/bin/bash
#SBATCH --job-name=orb_ml
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/orbml_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/orbml_%a_%j.err
#SBATCH --array=1-3111%200

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4

CASFILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
    /pc2/users/h/hpcmual/activeml/scripts/orbital_ml_commands.txt)

if [ -z "$CASFILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "Job $SLURM_ARRAY_TASK_ID: $CASFILE at $(date)"
python3 ~/activeml/scripts/casci_orbital_ml.py $CASFILE
echo "Done at $(date)"
