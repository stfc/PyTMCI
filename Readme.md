# PyTMCI
#### David Posthuma de Boer (ISIS, STFC, UKRI)

A python package to predict mode frequencies and growth rates for transverse,
coherent bunched beam instabilities including mode coupling. Python files
contain the code, and the jupyter notebooks (separate to this repository)
contain usage examples.


## Code Description
The code is spread across four files:
* `vlasov.py` contains the base class and general code structure
* `airbag_methods.py` contains methods specific for a longitudinal
  airbag model [1].
* `nht_methods.py` contains methods specific for a longitudinal
  nested airbag model (NHT) [2].
* `laguerre_methods.py` contains methods specific for the arbitrary
  longitudinal model, by making a series expansion with Laguerre
  polynomials.

The code is largely described in `vlasov.py`, whilst the implementation
details are contained in the other files. This helps to keep the code tidy,
but also allows the code to be compiled with Numba, if the user has Numba
installed; python methods are used otherwise.


## Installation

### User Automatic Set-Up

It is recommended to install PyTMCI into a virtual environment to maintain
correct dependency versions. Afterwards, the package can be installed using
pip with the command

```
python -m pip install pyTMCIVlasov
```

By default, this will install a version of the code that does not use Numba
for JIT compilation. To speed up the code, manually install Numba and Numba-scipy
with the commands

```
python -m pip install numba
python -m pip install git+https://github.com/numba/numba-scipy.git
```

PyTMCI will then automatically compile the code where it can.

__Notes__
1. The name PyTMCI is not available on pypi, as it is too similar to another package,
   so PyTMCIVlasov was used instead. 

1. numba-scipy is currently installed with a github url due to a dependency issue with
   the version on pypi. An ordinary pip install command will be used when the version
   on pypi is updated.


### Developer and Manual Set-Up

If the automatic set-up does not work, or to develop the code the package can
be installed manually. These instructions are given for Linux, but the
process should be similar for other operating systems. Before proceeding,
ensure that Python and git are installed on your system and can be run on a
terminal. Afterwards, open a terminal and follow these instructions

1. Clone this repository

1. Navigate to the root directory of this repository from a terminal

1. Create a new virtual environment
  `python -m venv ~/.venvs/pyTMCI`

1. Activate the new virtual environment
  `source ~/.venvs/pyTMCI/bin/activate`

1. Install the basic package and dependencies (no JIT)
  `python -m pip install -e .`

1. (Optional) If you would like use JIT compilation to speed up PyTMCI
   (with Numba), also run
   ```
   python -m pip install numba
   python -m pip install git+https://github.com/numba/numba-scipy.git
   ```

1. (Optional) If you would like to run the unit tests, also run
   `python -m pip install -e '.[tests]'`

1. (Optional) If you would like to use PyTMCI from Jupyter lab or notebook, 
   then install a Jupyter kernel for your new virtual environment
   `python -m pip install ipykernel`, then 
   `python -m ipykernel install --user --name=PyTMCI`
   This can then be chosen from within the jupyter interface.

1. (Optional) Install some useful packages for using PyTMCI (e.g. matplotlib,
   joblib)
   `python -m pip install -e '.[playground]'`

Many of these steps have been put into shell script, which can be run on Linux
by making sure it is executable
```
chmod +x setup_dev.sh
```
and running it with
```
./setup_dev.sh
```


## User Instructions
For instructions, see the example notebooks held separately. 


## Developer Instructions

### Running Unit Tests
First, run the manual set-up instructions and install the test dependencies, and
if possible the JIT compilation dependencies. First activate the PyTMCI virtual
environment with

```
source ~/.venvs/bin/activate
```

and then open a terminal and navigate to the repository's root directory. From
there make sure that the run_tests.sh script is executable with

```
chmod +x run_tests.sh
```

and run the all the unit tests with

```
./run_tests.sh
```

Individual unit tests can also be run by navigating to the `tests` directory and
running the command

```
python -m unittest FILENAME.py
```

where `FILENAME.py` is one of the python test files (e.g. `test_NHT.py`).


### Updating Dependencies

Dependencies are minimal and a typical scientific python stack will be able to
run the base package without issue. `pyproject.toml` contains pinned
versions, but it is possible there are compatible updates which would improve
performance. To test newer versions, update the dependencies in your virtual
environment with the following steps

1. Activate the python environment
   `source venv/bin/activate`

3. Upgrade dependencies
   `python -m pip install --upgrade DEPENDENCYNAME`

Afterwards run the unit tests. If performance has not degraded and the tests
pass, then please submit a pull request or make an issue.



## References

[1] A. W. Chao, Physics of Collective Beam Instabilities in High Energy
Accelerators, 1st ed. John Wiley & Sons, 1993. [Online]. Available:
http://www.slac.stanford.edu/%7Eachao/wileybook.html

[2] A. Burov, ‘Nested head-tail Vlasov solver’, Phys. Rev. ST Accel. Beams,
vol. 17, no. 2, p. 021007, Feb. 2014, doi: 10.1103/PhysRevSTAB.17.021007.

