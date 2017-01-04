from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

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
    install_requires=["keras>=1.1.0", "noise"],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    classifiers=(
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python"
    ))
