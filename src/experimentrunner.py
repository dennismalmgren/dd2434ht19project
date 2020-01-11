from dataloaders.newsgroupdatasetloader import NewsGroupDatasetLoader
import gin
import data_utils
from svm_original import SVM as SVM_original
from svm import SVM
import numpy as np
from kernels import ClusterKernel
import matplotlib.pyplot as plt
from markov_random_walk import MRW
import random
"""
# Import TSVM
import importlib
methods = importlib.import_module("semisup-learn.methods")
from methods.scikitTSVM import SKTSVM as tsvm

import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
"""

@gin.configurable
def train_tsvm(dataset_loader, num_labels=16, num_iter=100):
    # The TSVM package requires the following target outputs:
    # 0, 1 = labels for labelled data
    # -1 = unlabelled data
    input_, output = dataset_loader.get_full_dataset()
    pos_label, neg_label = 0, 1
    input_, output = data_utils.construct_one_vs_all(
        input_,
        output,
        primary_target=pos_label,
        pos_label=pos_label,
        neg_label=neg_label)

    y_train = np.array([-1] * len(output))
    print(input_.shape)
    random_labeled_points = (
        random.sample(list(np.where(output == pos_label)[0]), num_labels // 2) +
        random.sample(list(np.where(output == neg_label)[0]), num_labels // 2))
    y_train[random_labeled_points] = output[random_labeled_points]

    tsvm_model = tsvm()
    tsvm_model.fit(input_,
                   y_train)
    print(f"Transductive SVM score = {tsvm_model.score(input_, output)}")


def get_data_up_to_limit(dataset_loader, data_limit):
    input_, output = dataset_loader.get_full_dataset()
    if data_limit != 0:
        input_ = input_[:data_limit, :]
        output = output[:data_limit]
    return input_, output


@gin.configurable
def train_svm(dataset_loader, test_points, data_limit=0):
    input_, output = get_data_up_to_limit(dataset_loader, data_limit)

    input_, output = data_utils.construct_one_vs_all(input_, output, 0)
    (input_train, input_test, output_train, output_test) = data_utils.split(
        input_, output, test_points)
    #Run svm
    svm = SVM()
    svm.give_training_data(input_train, output_train)
    svm.train()

    svm.give_test_data(input_test, output_test)
    svm.analyze()

def random_walk_experiment():
    datasetLoader = NewsGroupDatasetLoader()
    datasetLoader.load_dataset()
    input, output = datasetLoader.get_full_dataset()
    input, output = data_utils.construct_one_vs_all(input, output, 0)

    n_labeled_points = 16

    print("Starting test for ", n_labeled_points, " datapoints.")
    results = 0
    for i in range(100):
        random_walk = MRW()

        print("Iteration #" + str(i))

        #Get the training indexes
        training_indexes = np.asarray(list(range(128)))
        training_targets_subset = []

        #Make sure that the data has both 1 and -1
        while 1 not in training_targets_subset or -1 not in training_targets_subset:
            training_indexes_subset = np.random.choice(
                training_indexes, n_labeled_points)
            training_targets_subset = output[training_indexes_subset]

        #Give the data to the random_walk
        random_walk.give_training_data(
            input[training_indexes_subset],
            training_targets_subset)

        # Adding the test values
        testing_indexes = np.asarray(list(range(128, 256)))
        random_walk.give_test_data(input[testing_indexes],
                                   output[128:256])

        # CLassify the dataset
        misclassification = random_walk.classify_dataset()
        results += misclassification
    y_results = results / 100
    plt.plot(y_results)
    plt.show()

def figure_2_experiment():
    dataset_loader = NewsGroupDatasetLoader()
    dataset_loader.load_dataset()
    kernel = ClusterKernel(kernel_name="STEP")
    input_, output = dataset_loader.get_full_dataset()
    input_, output = data_utils.construct_one_vs_all(input_, output, 0)

    x_results = [2,4,8,16,32,64,128]
    y_results = []
    for n_labeled_points in x_results:
        results = 0
        for i in range(20):
            print("Starting test for", n_labeled_points, "datapoints.", "Iteration #"+str(i+1))

            solution_found = False

            kernel_fun = kernel.kernel(input_)

            while not solution_found:

                #Send the data and unlabeled data (testing is analysed as unlabeled data)
                svm = SVM()
                svm.set_kernel(kernel.k)

                #Get the training indexes
                training_indexes = np.asarray(list(range(256)))
                training_targets_subset = []

                #Make sure that the data has both 1 and -1
                while 1 not in training_targets_subset or -1 not in training_targets_subset:
                    training_indexes_subset = np.random.choice(training_indexes, n_labeled_points)
                    training_targets_subset = output[training_indexes_subset]

                #Give the data to the SVM
                svm.give_training_data(training_indexes_subset, training_targets_subset)

                #Train the SVM.
                svm.train()

                solution_found = svm.solution_found

            #Send the indexes of labeled testing data and the labels
            testing_indexes = np.asarray(list(range(256, 256+128)))

            svm.give_test_data(testing_indexes, output[256:256+128])

            misclassification = svm.analyze()

            results += misclassification

        y_results.append(results/20)
    print(y_results)
    plt.plot(x_results, y_results)
    plt.show()

@gin.configurable
class ExperimentRunner:
    def __init__(self, experiment='single_run_newspaper'):
        self.experiment = experiment

    def RunExperiment(self):
        if self.experiment == 'single_run_newspaper':
            #Now we can load the data.
            dataset_loader = NewsGroupDatasetLoader()
            dataset_loader.load_dataset()

            train_svm(dataset_loader)
        elif self.experiment == 'figure_2':
            figure_2_experiment()
        elif self.experiment == 'random_walk':
            random_walk_experiment()
        elif self.experiment == 'transductive_svm':
            dataset_loader = NewsGroupDatasetLoader()
            dataset_loader.load_dataset()
            train_tsvm(dataset_loader)

        #Todo: Construct kernel for each type of kernel function.
        #Todo: Average test error over 100 random selections of labelled points, ranging from 2->128.
        #reuse the same kernel.
        #(ie 100 runs with 2, 100 runs with 4, 8, 16, 32, 64, 128)
        #Todo: Visualize and evaluate eigenvalue sizes also.
