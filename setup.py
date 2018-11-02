from setuptools import setup, find_packages

import re
import os

DIR = os.path.dirname(os.path.realpath(__file__))
INIT_FILE = os.path.join(DIR, 'doraemon', '__init__.py')

with open(INIT_FILE, 'r') as f:
    s = f.read()
    VERSION = re.findall(r"__version__\s*=\s*['|\"](.+)['|\"]", s)[0]

setup(name='Doraemon',
      version=VERSION,
      author='Guzzii',
      license='MIT',
      author_email='guzzii316@gmail.com',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.14.0',
          'pandas>=0.23.0',
          'cx_Oracle',
          'scikit-learn>=0.20.0',
          'scipy',
          'progressbar2',
      ],)
