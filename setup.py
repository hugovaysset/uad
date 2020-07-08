import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uad",
    version="0.0.1",
    author="Hugo Vaysset",
    description="Utils for anomaly detection using deep learning",
    packages=["MNIST", "models"],
    python_requires=">=3.6",
)
