from setuptools import setup, find_packages

setup(
    name='GraphScrDom',
    version='0.1.0',
    description='GraphScrDom',
    author='Xin Ma',
    author_email='xin.ma@ufl.edu',
    url='https://github.com/XinBiostats/GraphScrDom',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==1.4.4',
        'POT==0.9.4',
        'scanpy==1.9.6',
        'scikit-learn==1.3.2',
        #'scikit-misc==0.1.4',
        'scipy==1.14.1',
        'torch==1.13.1',
        'torch-geometric==2.5.2',
        'tqdm==4.64.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
