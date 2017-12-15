from setuptools import setup

setup(name='sqlplus',
      version='0.1.18',
      description='data work tools',
      url='https://github.com/nalssee/sqlplus.git',
      author='nalssee',
      author_email='kenjin@sdf.org',
      license='MIT',
      packages=['sqlplus'],
      # Install statsmodels manually with conda install
      install_requires=[
          'pypred',
          'sas7bdat',
          'xlrd'
      ],
      zip_safe=False)
