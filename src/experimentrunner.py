from dataloaders.newsgroupdatasetloader import NewsGroupDatasetLoader
import gin
import data_utils
from svm import SVM
import numpy as np
from kernels import ClusterKernel
import matplotlib.pyplot as plt


@gin.configurable
def train_svm(datasetLoader, test_points, data_limit=0):
    input, output = datasetLoader.get_full_dataset()
    if data_limit != 0:
        input = input[:data_limit, :]
        output = output[:data_limit]

    input, output = data_utils.construct_one_vs_all(input, output, 0)
    (input_train, input_test, output_train, output_test) = data_utils.split(
        input, output, test_points)
    #Run svm
    svm = SVM()
    svm.give_training_data(input_train, output_train)
    svm.train()

    svm.give_test_data(input_test, output_test)
    svm.analyze()


@gin.configurable
class ExperimentRunner:
    def __init__(self, experiment='single_run_newspaper'):
        self.experiment = experiment

    def RunExperiment(self):
        if self.experiment == 'single_run_newspaper':
            #Now we can load the data.
            datasetLoader = NewsGroupDatasetLoader()
            datasetLoader.load_dataset()

            train_svm(datasetLoader)
        elif self.experiment == 'figure_2':
            datasetLoader = NewsGroupDatasetLoader()
            datasetLoader.load_dataset()
            kernel = ClusterKernel()
            input, output = datasetLoader.get_full_dataset()
            input, output = data_utils.construct_one_vs_all(input, output, 0)

            x_results = [2, 4, 8, 16, 32, 64, 128]
            y_results = []

            for n_labeled_points in x_results:
                print("Starting test for", n_labeled_points, "datapoints.")
                results = 0
                for i in range(100):
                    print("Iteration #" + str(i))

                    kernel_fun = kernel.kernel(input)
                    svm = SVM()

                    #Send the data and unlabeled data (testing is analysed as unlabeled data)
                    svm.set_kernel(kernel_fun)

                    #Get the training indexes
                    training_indexes = np.asarray(list(range(128)))
                    training_targets_subset = []

                    #Make sure that the data has both 1 and -1
                    while 1 not in training_targets_subset or -1 not in training_targets_subset:
                        training_indexes_subset = np.random.choice(
                            training_indexes, n_labeled_points)
                        training_targets_subset = output[
                            training_indexes_subset]

                    #Give the data to the SVM
                    svm.give_training_data(training_indexes_subset,
                                           training_targets_subset)

                    #Train the SVM.
                    svm.train()

                    #Send the indexes of labeled testing data and the labels
                    testing_indexes = np.asarray(list(range(128, 256)))
                    svm.give_test_data(testing_indexes, output[128:256])
                    misclassification = svm.analyze()
                    results += misclassification
                y_results.append(results / 100)
            plt.plot(x_results, y_results)
            plt.show()

            #Todo: Construct kernel for each type of kernel function.
            #Todo: Average test error over 100 random selections of labelled points, ranging from 2->128.
            #reuse the same kernel.
            #(ie 100 runs with 2, 100 runs with 4, 8, 16, 32, 64, 128)
            #Todo: Visualize and evaluate eigenvalue sizes also.
