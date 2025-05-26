from setuptools import setup, find_packages

setup(
    name='bayesian-gp-cvloss',
    version='0.1.2',
    author='Shifa Zhong',
    author_email='sfzhong@tongji.edu.cn',
    description='A Python package for Gaussian Process Regression with hyperparameter optimization using Hyperopt and cross-validation, focusing on optimizing cross-validated loss.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Shifa-Zhong/bayesian-gp-cvloss',
    packages=find_packages(),
    install_requires=[
        'gpflow>=2.0.0',
        'hyperopt>=0.2.0',
        'scikit-learn>=0.23.0',
        'pandas>=1.0.0',
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
) 