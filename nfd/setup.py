from setuptools import setup

setup(
    name="nfd",
    py_modules=["neural_field_diffusion", "triplane_decoder"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
