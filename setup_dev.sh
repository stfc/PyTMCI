python -m venv ~/.venvs/PyTMCI
source ~/.venvs/PyTMCI/bin/activate
python -m pip install -e .
python -m pip install numba
python -m pip install git+https://github.com/numba/numba-scipy.git
python -m pip install -e '.[tests]'
deactivate
