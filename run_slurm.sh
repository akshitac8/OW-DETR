#!/bin/bash
#
#sBATCH --job-name=owdetr
#SBATCH --gpus=16
#SBATCH --nodes=2
#SBATCH --time=3-00:00:00
#

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh berz deformable_detr 16 configs/OWOD_new_split.sh
