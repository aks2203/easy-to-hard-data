from setuptools import setup, find_packages


setup(
    name='easy_to_hard_data',
    version='0.1',
    license='MIT',
    author="Avi Schwarzschild",
    author_email='avi1@umd.edu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/aks2203/easy-to-hard-data',
    keywords='easy to hard datasets',
    install_requires=[
          'torch',
      ],

)
