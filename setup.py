#!/usr/bin/env python
# coding=utf-8

"""The setup script."""

import ast

from setuptools import find_packages, setup  # type: ignore

with open("ece_loss/__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# Get the long description from the README file
with open("README.rst", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


setup(
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["torch~=1.9.0", "setuptools~=58.0.4"],
    extras_require={
        "dev": ["numpy~=1.21.2", "pytest", "pylint", "black", "isort", "tox"],
        "example": ["pillow~=8.4.0", "matplotlib~=3.4.3"],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="ece_loss",
    name="ece_loss",
    packages=find_packages(include=["ece_loss", "ece_loss.*"]),
    url="https://github.com/NKI-AI/ece_loss",
    version=version,
    zip_safe=False,
)
