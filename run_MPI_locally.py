#!/usr/bin/env python
'''
run example_case_HPC.py locally on n_pp processors
'''
__author__ = "Lanzilao Luca"
__date__ = "January 17, 2020"

from IPython import get_ipython

ip = get_ipython()
n_pp = '2'
ip.run_cell('!mpiexec -n '+n_pp+' python example_case_HPC.py')