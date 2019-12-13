#Todo: Andree will implement SVM

"""
Module containing the implementation of svm classifier.
"""
import gin
from kernels import ClusterKernel
from utils import LINEAR


@gin.configurable
class SVM:
    """Support vector machine with various kernels."""

    def __init__(self,
                 kernel_name=LINEAR,
                 degree=None,
                 num_labelled_points=None):
        self.kernel = ClusterKernel(
            kernel_name=kernel_name,
            degree=degree,
            num_labelled_points=num_labelled_points)

    def train(self, data):
        pass

    def predict(self, data):
        pass
