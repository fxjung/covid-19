from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="covid-19",
    version="0.1",
    description="covid-19",
    # long_description=long_description,
    # url='',
    # keywords='food recipes nutrition cooking',
    # classifiers=[
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3 :: Only',
    #     'Development Status :: 4 - Beta'
    # ],
    zip_safe=False,
    # packages=find_packages(exclude=['tests']),
    install_requires=[
        "numpy>=1.16.2",
        "xlrd>=1.2.0",
        "pandas>=1.0.2",
        "black",
        "jupyter",
        "matplotlib>=3.2.1",
        "seaborn>=0.10.0",
        "scipy>=1.4.1",
        "jupytext",
        "pre-commit>=2.2.0",
        "numba",
    ],
    author="Felix Jung",
    author_email="felix.jung@ds.mpg.de",
)
