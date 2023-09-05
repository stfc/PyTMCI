python -m venv ~/.venvs/PyTMCI
source ~/.venvs/PyTMCI/bin/activate
python -m pip install -e .
python -m pip install -e '.[jit]'
python -m pip install -e '.[tests]'
deactivate
