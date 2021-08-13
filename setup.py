#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install

with open("requirements.txt") as f:
    DEPENDENCIES = f.read().splitlines()

setup(
    name="ARLAffPoseDatasetUtils",
    packages=find_packages(),
    version='0.0.1',
    author="Aidan Keaveny",
    license="MIT",
    description="Package used to post-process dataset with ground truth object segmentation masks and 6-DoF pose",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=DEPENDENCIES,
    python_requires=">=3.6",
)
