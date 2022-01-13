# neutrino_oscillations
This program used various minimisation techniques to find the optimum parameters to fit predicted data to experimental data about a neutrino oscillation experiment.

Before using the program make sure you have the following Python 
libraries installed:
- Numpy
- Matplotlib
- TQDM

This repository contains the following Python files:
- `project1_initial_plots.py`
- `project1_nll_variation.py`
- `project1.py`
- `project1_cross_section.py`
These files correspond to different parts of the project.

## Initial Plots

Open `project1_initial_plots.py` and run it. This file should give the 
following graphs in order:
- experimental data histogram
- unoscillated data histogram
- probability function with initial parameters
- energy distribution with initial parameters
- probability function with varied parameters
- energy distribution with varied parameters
To access the next graph, close the matplotlib window.

## NLL Variation

Open `project1_nll_variation.py` and run it. This file will first give 
a graph of NLL variation with theta, followed by a graph of NLL 
variation with Delta m^2. Once the second graph window is closed, it 
will print, to the terminal, the optimised values of the parameters 
and their errors, from parabolic minimisaton. Please note, `dm` 
corresponds to Delta m^2.

## Newton's Method with 2 parameters

Open `project1.py` and run it. This file will output the optimised 
values of theta and Delta m^2, with their errors, obtained using
Newton's method.

## Rate of Change of Cross-Section

Open `project1_cross_section.py` and run it. It is very important that
you have installed tqdm because this is used as a progress bar for the
MCMC method. The program will first output the optimised parameters
with their errors obtained from Newton's method. It will then run the
Metropolis algorithm for 0 < T < 3 with step-size -0.5 and print the
values and errors to the terminal (including the NLL). It will run 
this again for 0 < T < 5 with step-size 1.
