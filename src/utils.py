"""
Module containing utility functions for, e.g. data cleaning and plots.
"""
import os


def get_src_folder():
    """
    Returns the path to the src folder based on the location of utils.py
    """
    scriptpath = os.path.realpath(
        __file__
    )  # This refers to the utils file which needs to be in src for this to work. Should be enough for this simple project.
    cur_dir = os.path.dirname(scriptpath)
    base_dir = os.path.dirname(cur_dir)
    datasets_dir = os.path.join(base_dir, 'src')
    return datasets_dir
