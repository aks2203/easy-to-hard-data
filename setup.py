from setuptools import setup
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, 'easy-to-hard-data'))
from version import __version__

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

print('version')
print(__version__)

setup(name='easy_to_hard_data',
      version="v0.1.4",
      description='Easy to Hard Data',
      url='https://aks2203.github.io/easy-to-hard-data',
      author='Avi Schwarzschild',
      keywords=['pytorch', 'generalization', 'machine learning'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      py_modules=['easy_to_hard_data'],
      python_requires='>=3.8.2',
      install_requires=[
        'numpy>=1.18.5',
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'tqdm>=4.60.0'
     ],
      license='MIT'
)
