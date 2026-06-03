#!/bin/bash
#SBATCH --job-name=fe2s2_single
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_single_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_single_%j.err
#SBATCH --nodelist=n2lcn0149

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MKL_THREADING_LAYER=GNU
export MKL_INTERFACE_LAYER=LP64
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.11/site-packages/block2.libs:$LD_LIBRARY_PATH

TESTFILE=$(head -1 /pc2/users/h/hpcmual/activeml/scripts/fe2s2_commands.txt)
echo "Testing: $TESTFILE"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
python3 ~/activeml/scripts/fe2s2_features.py $TESTFILE
echo "Finished: $(date)"
