# binaries

`binaries` is a set of Python notebooks to calculate the evolution of wide stellar binaries as one or both of the component stars lose mass and / or undergo an associated velocity kick on their way to becoming white dwarfs. The code was written by Nadia Zakamska. This code is written for Python 3, and it depends on a handful of standard mathematical and plotting packages. If you use `binaries` in your research, please cite Zakamska and Hwang 2024 (in prep., will be updated as soon as available). 

The file `orbital.py` is the necessary set of helper functions used throughout the notebooks. It should be put in the same directory as the notebooks (the top section of the notebooks contains the import statement for this file). The file `test_orbital.py` shows usage of some of the functions in `orbital.py`. 

`public_testing_hamiltonian.ipynb` notebook presents tests of our secular evolution equations. We test against analystical solutions and direct numerical integration of orbits using `rebound` available at https://rebound.readthedocs.io/en/latest/. `rebound` does not cover the full scientifically interesting case of both mass loss and acceleration, so we can only test our equations in the regime of constant mass. 
