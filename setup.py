from setuptools import setup

setup(name='KBFC',
      version='0.1.0',
      description='KBase File Validation Classification System',
      #url='http://github.com/storborg/funniest',
      author='Brandon Feng',
      author_email='brf39@cornell.edu',
      license='MIT',
      packages=['KBFC'],
      install_requires = [
          'numpy', 'matplotlib', 'torch', 'pandas', 'sklearn', 'os', 'gzip', 'random', 're', 'xgboost',
          'pickle', 'seaborn', 'scikitplot', 'time', 'tqdm', 'inspect', 'warnings',
      ],
      zip_safe=False)