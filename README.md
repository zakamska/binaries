# binaries

`binaries` is a set of Python notebooks to calculate the evolution of wide stellar binaries as one or both of the component stars lose mass and / or undergo an associated velocity kick on their way to becoming white dwarfs. The code was written by Nadia Zakamska. This code is written for Python 3, and it depends on a handful of standard mathematical and plotting packages, plus we test against `rebound` available at https://rebound.readthedocs.io/en/latest/. If you use `binaries` in your research, please cite Zakamska and Hwang 2024 (in prep., will be updated as soon as available). 

The file `orbital.py' is the necessary set of helper functions used throughout the notebooks. The file `test_orbital.py' shows usage of some of the functions in `orbital.py'. 

`public_testing_hamiltonian' notebook presents tests of our secular evolution equations. `Rebound' does not cover the full scientifically interesting case of mass loss and acceleration, so we can only test our equations in the regime of constant mass. We also use analytical solutions where available. 
