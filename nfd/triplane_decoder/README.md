# HierarchicalMLP

Setup:
```
conda env create -f environment.yml
conda activate 3d_mlp

cd inside_mesh
python setup.py build_ext --inplace
cd ..
```

If you want to use your own 3D meshes for 3D fitting, first preprocess the data:

```
python generate_3d_dataset.py --input=data/thai_statue.ply --output=data/thai_statue.npy
```

If you want to visualize 3D shapes: Download ChimeraX from UCSF. Drag the .mrc file into the pane, set the threshold to about 0.5, and change the mode to 'surface'. For the best quality, change 'step' to 1.
# HierarchicalMLP
