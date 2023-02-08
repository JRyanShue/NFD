

# input: cuda devices to run on, triplane range to optimize
cuda_devices=$1
data_dir=$2
triplane_range=$3

echo $cuda_devices
echo $data_dir
echo $triplane_range

# split triplane range among the cuda devices

# for each sub-range, run the python script
# CUDA_VISIBLE_DEVICES=7 python triplane_diffusion/train_generalize.py \
# --data_dir ../data/preprocessed/02958343 --batch_size 10 --load_ckpt_path \
# /home/jrshue/neural-field-diffusion/ckpts_500shapeAD_edr0.01_128res/model_epoch_1999_loss_0.19445328414440155.pt \
# --checkpoint_path ckpts_triplanes_600-700_500to100_shape_edr0.01_128res --training_subset_size 500 \
# --skip_number 100 --subset_size 100 --edr_val 0.01 --save_every 2000 --use_tanh &