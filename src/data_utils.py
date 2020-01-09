import os
import sklearn.model_selection as skm
"""
Helper functions for manipulating files and directories (might need to be moved?)
"""
"""
Helper function that splits input and output into to sets where the test_count parameter indicates how many
samples to put in the second set from input and output.
returns input_first_set, input_second_set, output_first_set, output_second_set
"""


def split(input, output, test_count):
    return skm.train_test_split(input, output, test_size=test_count)


"""
Adjusts target labels so that the sought after class has output label pos_label
and every other item shares the same label of neg_label
"""


def construct_one_vs_all(input,
                         output,
                         primary_target,
                         pos_label=1,
                         neg_label=-1):
    output[output != primary_target] = neg_label
    output[output == primary_target] = pos_label
    return input, output


"""
Returns the path to the datasets folder based on the location of utils.py
"""


def get_datasets_folder():
    scriptpath = os.path.realpath(
        __file__
    )  # This refers to the utils file which needs to be in src for this to work. Should be enough for this simple project.
    cur_dir = os.path.dirname(scriptpath)
    base_dir = os.path.dirname(cur_dir)
    datasets_dir = os.path.join(base_dir, 'datasets')
    return datasets_dir
