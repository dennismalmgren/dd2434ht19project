import os


def split(input, output, test_count):
    return skm.train_test_split(input, output, test_size=test_count)

"""
Helper functions for manipulating files and directories (might need to be moved?)
"""
"""
Returns the path to the datasets folder based on the location of utils.py
"""
def get_datasets_folder():
    scriptpath = os.path.realpath(__file__) # This refers to the utils file which needs to be in src for this to work. Should be enough for this simple project.
    cur_dir = os.path.dirname(scriptpath)
    base_dir = os.path.dirname(cur_dir)
    datasets_dir = os.path.join(base_dir, 'datasets')
    return datasets_dir