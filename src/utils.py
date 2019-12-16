import os
import sklearn.model_selection as skm

"""
Module containing constants and utility functions for, e.g. data cleaning and plots.
"""

LINEAR = 'linear_kernel'
STEP = 'step'
LINEAR_STEP = 'linear_step'
POLY = 'poly'
POLY_STEP = 'poly_step'

"""
Returns the path to the src folder based on the location of utils.py
"""
def get_src_folder():
    scriptpath = os.path.realpath(__file__) # This refers to the utils file which needs to be in src for this to work. Should be enough for this simple project.
    cur_dir = os.path.dirname(scriptpath)
    base_dir = os.path.dirname(cur_dir)
    datasets_dir = os.path.join(base_dir, 'src')
    return datasets_dir