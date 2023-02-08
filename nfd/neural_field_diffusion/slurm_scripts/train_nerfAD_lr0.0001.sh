#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=viscam5
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="NerfAD_lr0.0001"
#SBATCH --output=NerfAD_lr0.0001-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=jryanshue@gmail.com
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# Environment setup
cd /viscam/projects/triplane-diffusion/neural-field-diffusion
source /sailhome/jrshue/miniconda3/etc/profile.d/conda.sh
conda activate nfd2

# Auto-decoder training
CUDA_VISIBLE_DEVICES=0 python triplane_fitting/nerf_fitting/train.py --data_dir /viscam/u/erchan/triplane-diffusion/cars_650/ryan_cars --subset_size 200 --save_every 50 --save_path 200shape_66image_nosampling_notanh_lr0.0001 --ray_batch_size 5000 --log_every 2 --val_every 10 --nouse_view_embed --lr 0.0001

# done
echo "Done"
