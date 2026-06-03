#!/bin/bash
#SBATCH --job-name=fe2s2_feat
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_feat_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_feat_%a_%j.err
#SBATCH --array=1-880%200

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4
export LD_PRELOAD=/pc2/users/h/hpcmual/activeml/lib/lapack_wrapper.so
export LD_LIBRARY_PATH=/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.23-GCC-12.3.0/lib:$LD_LIBRARY_PATH

CASFILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
    /pc2/users/h/hpcmual/activeml/scripts/fe2s2_commands.txt)

if [ -z "$CASFILE" ]; then exit 0; fi

echo "Task $SLURM_ARRAY_TASK_ID: $CASFILE"
echo "Started: $(date)"
python3 ~/activeml/scripts/fe2s2_features_casscf.py $CASFILE
echo "Finished: $(date)"
