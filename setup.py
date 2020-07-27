import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uad",
    version="0.0.1",
    author="Hugo Vaysset",
    author_email="hugo.vaysset@polytechnique.edu",
    description="Models and utils for unsupervised anomaly detection using keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hugovaysset/uad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras_preprocessing']
)