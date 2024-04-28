from setuptools import setup, find_packages

setup(name='lt_toolbox',
      version='0.1.0',
      description='Lagrangian Trajectories Toolbox',
      url='http://github.com/oj_tooth/lt_toolbox',
      author='Ollie Tooth',
      author_email='oliver.tooth@seh.ox.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=["numpy", "scipy", "polars", "xarray", "plotly", "numba"]
      )
