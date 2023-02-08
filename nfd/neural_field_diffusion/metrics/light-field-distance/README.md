# Light Field Distance metric
Original repo: [link](https://github.com/Sunwinds/ShapeDescriptor)

## Note
The code was converted to be able to use LFD metric (distance between two 
descriptors) that will compare visual appearance between ground truth mesh and 
retrieved mesh.

## This fork
The original repository was adapted partially to run on Linux. Only `LightField` 
was changed so it can be used through docker without any dependency. Underneath,
the container uses OSMesa for headless rendering. 


## Requirements
- `pip install trimesh`

## Installation
```bash
pip install light-field-distance
```

or 

```
python setup.py install
```

No need to explicitly install anything.

## Usage
```python
from lfd import LightFieldDistance
import trimesh

# rest of code
mesh_1: trimesh.Trimesh = ...
mesh_2: trimesh.Trimesh = ...

lfd_value: float = LightFieldDistance(verbose=True).get_distance(
    mesh_1.vertices, mesh_1.faces,
    mesh_2.vertices, mesh_2.faces
)
```
The script will calculate light field distances 
[[1]](http://www.cs.jhu.edu/~misha/Papers/Chen03.pdf) between two shapes. 
Example usage:
```python
from lfd import LightFieldDistance
import trimesh

# rest of code
mesh_1: trimesh.Trimesh = trimesh.load("lfd/examples/cup1.obj")
mesh_2: trimesh.Trimesh = trimesh.load("lfd/examples/airplane.obj")

lfd_value: float = LightFieldDistance(verbose=True).get_distance(
    mesh_1.vertices, mesh_1.faces,
    mesh_2.vertices, mesh_2.faces
)
```
The lower the metric's value, the more similar shapes are in terms of the visual
appearance

## How does it work
The `lfd.py` is a proxy for the container that install all the dependency necessary
to run a C code. The code performs calculation of Zernike moments and other
coefficients that are necessary to calculate the distance (`3DAlignment` program).
Then, these coefficients are saved and run by the `Distance` program that calculated the
Light Field Distance. It prints out the result and the stdout from the printing
is handled by the python script.

If an image for the C code is not found, it builds one. The operation is performed
once and it takes a while to finish it. After that, the script runs the necessary 
computations transparently.

## Contribution
For anyone interested in having a contribution, these are things to be done. 
Due to the time constraints, I'm not able to do these on my own:
- [ ] retrieve calculating coefficients from renders to be returned by a method
- [ ] bind C code with pybind11 to allow direct computation from the python code
    without any Docker dependency

## How am I sure that it works as supposed?
I checked descriptor artifacts from the original implementation and compared with results in the docker through md5sum

