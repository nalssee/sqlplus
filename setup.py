# import os
from setuptools import setup

# dir = os.path.dirname(os.path.realpath(__file__))

# with open(os.path.join(dir, 'requirements.txt')) as f:
#     required = f.read().splitlines()

setup(name='sqlplus',
      version='0.1.12',
      description='data work tools',
      url='https://github.com/nalssee/sqlplus.git',
      author='nalssee',
      author_email='kenjin@sdf.org',
      license='MIT',
      packages=['sqlplus'],
      # There are important others like pandas, statsmodels
      # but those can't be properly installed using pip in Windows
      # So you must install those manually using Anaconda navigator TT
      install_requires=[
          'pypred',
          'sas7bdat',
          'xlrd'
      ],
      zip_safe=False)
