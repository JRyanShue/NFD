#!/bin/bash
# Download models (ddpm and decoder) for the cars, chairs, and planes classes of ShapeNet reported in our paper.

cd models

# DDPM checkpoints
wget https://imt-public-datasets.s3.amazonaws.com/ddpm_ckpts/cars/ddpm_cars_405k.zip && unzip ./*.zip -d cars && rm ./*.zip
wget https://imt-public-datasets.s3.amazonaws.com/ddpm_ckpts/chairs/ddpm_chairs_200k_v2.zip && unzip ./*.zip -d chairs && rm ./*.zip
wget https://imt-public-datasets.s3.amazonaws.com/ddpm_ckpts/planes/ddpm_planes_220k.zip && unzip ./*.zip -d planes/ddpm_planes_ckpts && rm ./*.zip

# Decoder checkpoints
wget https://imt-public-datasets.s3.amazonaws.com/decoder_ckpts/car_decoder.pt -P cars/
wget https://imt-public-datasets.s3.amazonaws.com/decoder_ckpts/chair_decoder.pt -P chairs/
wget https://imt-public-datasets.s3.amazonaws.com/decoder_ckpts/plane_decoder.pt -P planes/

# Statistics
wget https://imt-public-datasets.s3.amazonaws.com/triplane_statistics/cars_triplanes_stats.zip && unzip ./*.zip -d cars/statistics && rm ./*.zip
wget https://imt-public-datasets.s3.amazonaws.com/triplane_statistics/chairs_triplanes_stats.zip && unzip ./*.zip -d chairs/statistics && rm ./*.zip
wget https://imt-public-datasets.s3.amazonaws.com/triplane_statistics/planes_triplanes_stats.zip && unzip ./*.zip -d planes/statistics && rm ./*.zip

cd ..