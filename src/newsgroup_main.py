# This file contains executable for the e level assignment. 
# It reads data based on experiment gin file and runs a set of kernels to produce output
# for a segment of the report.
# Not sure yet if we should split mains per dataset or not, etc.

import data_utils
import utils
import argparse
import gin
from svm import SVM
from dataloaders.newsgroupdatasetloader import NewsGroupDatasetLoader
import kernels

def parse_args():
    parser = argparse.ArgumentParser(description='Perform an experiment')
    parser.add_argument(
        '--ginfile',
        default=
        utils.get_src_folder() + '/newsgroup_main_template.gin')
    #To be added once we manage to run experiments:
    #parser.add_argument('--experiment_folder', default='./experiments/')
    #parser.add_argument('--experiment_name', default='experiment')
    return parser.parse_args()

@gin.configurable
def train_fixed_test_point_count(datasetLoader, test_points):
    input, output = datasetLoader.get_full_dataset()
    input, output = data_utils.construct_one_vs_all(input, output, 0)
    (input_train, input_test, output_train, output_test) = data_utils.split(input, output, 987)
    #Run svm
    svm = SVM()
    svm.give_training_data(input_train, output_train)
    svm.train()

    svm.give_test_data(input_test, output_test)


def main():
    args = parse_args()
    #This line sets up constructor arguments for kernels etc based on the contents of the gin file.
    gin.parse_config_file(args.ginfile)

    #Now we can load the data.
    datasetLoader = NewsGroupDatasetLoader()
    datasetLoader.load_dataset()
    
    train_fixed_test_point_count(datasetLoader)

  
    #Prints parameters used during the run
    print(gin.operative_config_str())

if __name__=="__main__":
    main()
