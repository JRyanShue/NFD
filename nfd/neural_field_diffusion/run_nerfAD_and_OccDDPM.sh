#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=viscam5
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:2

#SBATCH --job-name="NerfAD_plus_OccDDPM"
#SBATCH --output=NerfAD_plus_OccDDPM-%j.out

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
CUDA_VISIBLE_DEVICES=0 python triplane_fitting/nerf_fitting/train.py --data_dir /viscam/u/erchan/triplane-diffusion/cars_650/ryan_cars --subset_size 200 --save_every 50 --save_path 200shape_66image_nosampling_notanh_2 --ray_batch_size 5000 --log_every 2 --val_every 10 --nouse_view_embed --load_ckpt_path /viscam/projects/triplane-diffusion/neural-field-diffusion/200shape_66image_nosampling_notanh_2/1665028303/ckpts/model_epoch_11_loss_0.014797789976000786.pt &

# DDPM training
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 24"
CUDA_VISIBLE_DEVICES=1 python scripts/image_train.py --data_dir /viscam/projects/triplane-diffusion/data/occupancy_triplanes/triplanes --in_out_channels 96 $MODEL_FLAGS $TRAIN_FLAGS

# done
echo "Done"
