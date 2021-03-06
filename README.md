# Some Plots on Covid-19
jupyter notebook that downloads publicly available data on the spread 
of Covid-19 and allows for easy creation of several plots.

**Note: I am not an expert in this field. There may be (severe) bugs or mistakes.
Do not use this for any actual decision making. The code is provided with no warranty whatsoever.**

## Prerequisites
- git
- Python 3.7
- pip

## Installation
```bash
git clone git@github.com:fxjung/covid-19.git
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
