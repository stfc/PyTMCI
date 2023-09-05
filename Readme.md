# Vlasov Solutions for Bunched Particle Beams
#### David Posthuma de Boer (ISIS, STFC) (david.posthuma-de-boer@stfc.ac.uk)

A collection of files for solving the Vlasov Equation for
bunched particle beams. Python files contain the code, and the
jupyter notebooks contain usage examples.

## Code Description
The code is spread across four files:
* `vlasov.py` contains the base class and should be imported
* `airbag_methods.py` contains methods specific for a longitudinal
  airbag model.
* `nht_methods.py` contains methods specific for a longitudinal
  nested airbag model.
* `laguerre_methods.py` contains methods specific for the arbitrary
  longitudinal model, by making a series expansion with Laguerre polynomials.

The code is largely described in vlasov.py, whilst the implementation
details are contained in the other files. This helps to keep the code
tidy, but also allows the code to be compiled with Numba, if the user
has Numba installed; python methods will be used otherwise.


## Running the Code
The Vlasov solvers are implemented as python files. These can be
imported and run within Jupyter notebooks. Dependencies are minimal
(see below) and a typical scientific python stack will be able to
run them without issue. The code will run on Linux or Windows, but
the set-up instructions might be slightly different. 

Experienced users can clone the repository, install the dependencies
and run the Benchmark notebook; otherwise instructions are given below. A
virtual environment is used to ensure that the same versions of
direct dependencies are used.


### Set-up from Scratch
Install Python and Git, then follow these steps:
1. Clone this repository
2. Navigate to repository directory from a terminal
3. Create a new virtual environment  
  `python -m venv venv`
4. Activate the virtual environment  
  Linux `source venv/bin/activate`  
  Windows `.\venv\Scripts\activate.bat` 
5. Install the requirements into the virtual environment  
  `python -m pip install -r requirements.txt`
6. Install a Jupyter kernel for your virtual environment  
  `python -m ipykernel install --user --name=vlasov`
7. Start Jupyter Lab  
  `python -m jupyter lab`
8. Open your web browser and navigate to the printed url
9. Open one of the Notebooks and Charge the Kernel to "vlasov"
10. Run the notebook

Subsequently the `vlasov` python kernel should be available from jupyter
by starting it as usual `jupyter lab`.

### Updating Dependencies
requirements.txt contains pinned versions (exact version numbers),
to try and make results reproducible between machines, but it is possible
there are compatible updates which would improve performance. To test
newer versions, update the dependencies in your virtual environment
with the following steps
1. Navigate to the repository directory in a terminal
2. Activate the python environment  
  Linux `source venv/bin/activate`  
  Windows `.\venv\Scripts\activate.bat` 
3. Upgrade dependencies  
`python -m pip install --upgrade DEPENDENCYNAME`

Afterwards run a set of benchmarks in the notebooks. If performance has
not degraded and results are accurate then submit a request for an
updated requirements file.
