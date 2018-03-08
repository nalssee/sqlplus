from setuptools import setup

setup(name='sqlplus',
      version='2.5.1',
      description='data work tools',
      url='https://github.com/nalssee/sqlplus.git',
      author='nalssee',
      author_email='kenjin@sdf.org',
      license='MIT',
      packages=['sqlplus'],
      # Install statsmodels manually with conda install
      install_requires=[
          'sas7bdat==2.0.7',
          'xlrd==1.1.0',
          'openpyxl==2.5.0',
          'psutil==5.4.3',
      ],
      # scripts=['bin/prepend', 'bin/fnguide', 'bin/xl2csv'],
      zip_safe=False)
