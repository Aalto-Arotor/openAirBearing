
[![PyPI Version](https://img.shields.io/pypi/v/openairbearing.svg)](https://pypi.org/project/openairbearing)
[![Unit tests](https://github.com/Aalto-Arotor/openAirBearing/actions/workflows/unittests.yml/badge.svg)](https://github.com/Aalto-Arotor/openAirBearing/actions/workflows/unittests.yml)
[![Test coverage](https://coveralls.io/repos/github/Aalto-Arotor/openAirBearing/badge.svg?branch=main)](https://coveralls.io/github/Aalto-Arotor/openAirBearing?branch=main)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aalto-Arotor/openairbearing/blob/main/LICENSE)

<p align="center">
    <img src="docs/OpenAirBearing.png" alt="OpenAirBearing logo" width="80%">
</p>

# OpenAirBearing: Open-Source Porous Gas Bearing Analysis
## Introduction

OpenAirBearing is a software package for modeling porous bearing performance.
It includes analytical solutions for simplified cases, finite-element-method-based framework for complex geometries, and a browser GUI for setting input parameters and displaying results.
The software can be used online at https://www.openairbearing.com with limited capability, and with full capability when used locally.

Supported features include:
- Bearing geometries:
    - Circular, rectangular, and annular thrust bearings
    - Infinitely long linear bearings and seals
- Calculation of load capacity, stiffness, air consumption, tilting moment, and shear force
- Sensitivity to air-gap variation (tilting, deformation, static shape errors)

## Mathematical modeling

The package provides analytical and numerical solutions of the Reynolds equation in one dimension for the most common porous gas bearing and seal geometries.
Analytical solutions assume ideal geometry, while numerical (finite element method) solutions consider uneven gap height, uneven permeability, and slip at the porous-gap interface.
Implements models from textbooks [1,2] and research publications [3,4].
The numerical models use the scikit-fem package for solving the air-gap domain: https://github.com/kinnala/scikit-fem.

## Installation

Python is required to use OpenAirBearing. You can install Python from the official website (https://www.python.org/).

OpenAirBearing can be installed using the Python package installer `pip` (https://pypi.org/project/pip/):

    pip install openairbearing[ui]

This includes solver, plotting, and UI dependencies. Additionally, a lightweight install without the UI is supported:

    pip install openairbearing


Alternatively, `git` can be used to clone the repository:
    
    git clone https://github.com/Aalto-Arotor/openAirBearing.git
    
This creates a folder *openairbearing* containing the OpenAirBearing source code.

## Quickstart

OpenAirBearing user interface can be started with:

    python openairbearing/app/app.py

This launches the Dash application, which can be accessed at:

    http://127.0.0.1:8050


## Examples

The `examples/` folder contains numbered scripts that highlight key features.

1. `ex00_run_app_local.py`  
    Print local app launch options (`openairbearing`, module run, Python API).

2. `ex01_circular_baseline.py`  
    Circular bearing baseline using analytic, FEM 1D, and FEM 2D solvers.

3. `ex02_circular_geometry_error_comparison.py`  
    Circular FEM 2D comparison for geometry error modes: none, linear, quadratic, saddle, tiltx, tilty.

4. `ex03_rectangular_velocity_comparison.py`  
    Rectangular FEM 2D full-Reynolds comparison for sliding velocities 0, 5, and 10 m/s.

5. `ex04_custom_geometry_function.py`  
    Rectangular FEM 2D comparison between default geometry and a custom geometry callback.

6. `ex05_experimental_comparison.py`  
    Compare model outputs against included experimental datasets for multiple test cases.

7. `ex06_circular_tilt_sweep.py`  
    Circular FEM 2D tilt sweep with a custom gap height function.

8. `ex07_journal_geometry_error_comparison.py`  
    Journal FEM 2D comparison for journal-specific geometry errors: none, conicity, and misalignment.

Run any example from the repository root:

     python examples/ex01_circular_baseline.py

## Contact
The software is being developed at the Arotor Lab at Aalto University, Finland. The main developer is Mikael Miettinen. Jalmari Lee contributed to the finite element solutions. Tom Gustafsson supported in the use of scikit-fem.

https://www.aalto.fi/en/department-of-energy-and-mechanical-engineering/aalto-arotor-lab

For any questions regarding the software, please contact mikael.miettinen@aalto.fi.

## Acknowledgements
This software has been developed as part of publicly funded research projects, including Business Finland projects Power Beyond (grant number 2534/31/2022) and BEST (grant number 1740/31/2025). Raine Viitala is acknowledged for his significant role in funding acquisition.

### References
[1] V. N. Constantinescu, Gas Lubrication, American Society of Mechanical Engineers, 1969. URL: https://archive.org/details/gaslubrication0000cons/

[2] F. Al-Bender, Air Bearings - Theory, Design and Applications, John Wiley & Sons, 2021. doi: https://doi.org/10.1002/9781118926444

[3] M. Miettinen, V. Vainio, R. Theska, R. Viitala, On the static performance of aerostatic elements, Precision Engineering 89 (2024) 1–10. doi:  https://doi.org/10.1016/j.precisioneng.2024.05.017.

[4] M. Miettinen, V. Vainio, R. Viitala, Aerostatic porous annular thrust bearings as seals, Tribology International 200 (2024) 110073. doi: https://doi.org/10.1016/j.triboint.2024.110073.
