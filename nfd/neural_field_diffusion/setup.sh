#!/bin/bash
# Run setup

mkdir modules
cd modules

# Kaolin-wisp install (https://github.com/NVIDIAGameWorks/kaolin-wisp.git)

git clone https://github.com/NVIDIAGameWorks/kaolin-wisp.git
cd kaolin-wisp

conda activate nfd
pip install --upgrade pip

sudo apt-get update
sudo apt-get install libopenexr-dev 

cd ..
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin  # Cloning kaolin
cd kaolin
python setup.py develop

cd ../kaolin-wisp
pip install -r requirements.txt
python3 setup.py develop
