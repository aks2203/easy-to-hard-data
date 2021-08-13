from setuptools import setup
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "easy-to-hard-data"))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="easy_to_hard_data",
      version="v1.0.0",
      description="Easy to Hard Data",
      url="https://aks2203.github.io/easy-to-hard-data",
      author="Avi Schwarzschild",
      keywords=["pytorch", "generalization", "machine learning"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      py_modules=["easy_to_hard_data", "easy_to_hard_plot"],
      python_requires=">=3.7",
      install_requires=[
        "chess==1.6.1",
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "reportlab==3.5.68",
        "seaborn>=0.11.0",
        "svglib==1.1.0",
        "torch>=1.7.0",
        "torchvision>=0.8.2",
        "tqdm>=4.60.0"],
      license="MIT")
