# numerical_qkd

This repository contains the numerical method for calculating the secret key rate
of QKD with the finite key analysis.

## Installation

This package uses and is written in Python 3.x

It is optional to use/install the MATLAB tools, but we use MATLAB and CVX 
because certain SDPs take too long to be solved.
Please install the following MATLAB tools:
- [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- [CVX for MATLAB](http://cvxr.com/)
- [cvxquad](https://github.com/hfawzi/cvxquad)

Optional (but desirable):
Install [MOSEK](https://www.mosek.com) solver for python.

### Installation steps

1. If you'd like to use the MATLAB tools, please install the tools described above.

2. Install the required python packages using: ```pip install -r requirements.txt```

3. Installation of Numerical QKD Python package can be done via setuptools >= 36.2.7: ```python setup.py install```

## Convex optimization tools

The convex optimization problems are solved using two disciplined convex
programming (DCP) tools: 
- **cvxpy** : www.cvxpy.org (Python)
- **cvx** :  www.cvxr.com (MATLAB)

## Usage

Please refer see the Jupyter notebooks in the ```examples``` folder to see
how you can use this library.
The simplest example is: ```simple_BB84_example.ipynb```.
To see how postselection is performed, see ```BB84_protocol.ipynb```.
