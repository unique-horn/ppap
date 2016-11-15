#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

install_requires = [x.strip() for x in requirements]

project_url = "https://github.com/"
project_url += "unique-horn/ppap"

setup(
    name="ppap",
    version="0.1.0",
    description="Pattern Producing Network layers for keras",
    long_description=readme,
    author="Unique Horn",
    author_email="",
    url=project_url,
    include_package_data=True,
    install_requires=install_requires,
    license="GPL v3",
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=(
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python"
    ),
    zip_safe=False)
