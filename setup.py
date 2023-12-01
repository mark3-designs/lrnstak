from setuptools import setup, find_packages

setup(
    name='lrnstak',
    version='0.1.0',
    description='A versatile machine learning framework for feature engineering and model training.',
    author='KWIQ Solutions, LLC',
    author_email='info@kwiqsolutions.com',
    packages=[
        'lrnstak'
    ],
    install_requires=[
        'scikit-learn',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
