# OOP with Python
- Python scripts based on [Platzi](https://platzi.com/c/enez/ "Platzi") [course](https://platzi.com/clases/poo-python/ "course") 💚
- Use Python 3
- Working on this, not all scripts uploaded

## Windows installation guide
1. [Download](https://www.python.org/downloads/ "Download") Python 3 for Windows
2. Run executable installer
3. Verify Python was installed
	- Open CMD (win + r, then type "cmd" without the quotes)
	- Type "python", you should get the info of the python version
	- exit() to quit
4. Verify pip was installed
	- Open CMD
	- Type "pip -V", if installed successfully, you should get the installation route
5. Add python path to environment variables

> To run python scripts, open CMD, move onto your script's directory and enter: 
>  python scriptName.py  // VSCode recommended to avoid this step

## Setting a virtual environment - Windows
    mkdir directoryName
    cd directoryName
    python -m venv env  // env is the name of the virtual environment
    env\Scripts\activate.bat
    pip install bokeh
## Setting a virtual environment - Linux system
    mkdir directoryName
    cd directoryName
    python[version] -m venv env
    source env/bin/activate
    pip install bokeh
