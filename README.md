# Some Plots on Covid-19

## Prerequisites
- git
- Python 3.7
- pip

## Installation
```bash
git clone git@gitlab.gwdg.de:fjung1/covid-19.git
cd covid-19
pip install .
pre-commit install
```

## Usage
```bash
jupyter notebook
```
Then execute all cells, data should be downloaded and plots created in the working directory.
If data download fails, possibly adjust the timestamp used to infer the download url to yesterday.

## Contributing
* Code style: black
* Don't push notebooks, 'pair with light script' instead.
* All input welcome! :)
