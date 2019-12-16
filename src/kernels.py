"""
Module containing the implementation of various cluster kernels.
"""
from functools import partial
import gin
from utils import LINEAR, LINEAR_STEP, STEP, POLY, POLY_STEP
import numpy as np

@gin.configurable
class ClusterKernel:
    def __init__(self,
                 kernel_name=LINEAR,
                 degree=None,
                 sigma=0,
                 cutoff_type = 'n_relative', #either n_relative or absolute
                 r = 10, #This defines cutoff for step, linear-step and poly-step.
                 p=2, #Power for poly-step under or equal to cutoff
                 q=2, #Power for poly-step over cutoff
                 num_labelled_points=None):
        """
        Generates a ClusterKernel of the type kernel_name.
        To compute the kernel for data X, call obj.kernel(X)
        where obj is the name of the ClusterKernel instance.

        Args:
            - degree: required if 'poly' kernel is selected
            - num_labelled_points: required if 'poly-step' kernel is selected
        """
        self.kernel_name = kernel_name
        self.degree = degree
        self.num_labelled_points = num_labelled_points
        self.kernel_func = self._get_kernel_func()

        self.kernel = partial(self._compute_kernel, self.kernel_func)

    def _get_kernel_func(self):
        kernel_name_mapping = {
            LINEAR: self._linear_tf,
            STEP: self._step_tf,
            LINEAR_STEP: self._linear_step_tf,
            POLY: self._poly_tf,
            POLY_STEP: self._poly_step_tf
        }
        selected_kernel = kernel_name_mapping[self.kernel_name]

        if self.kernel_name == POLY:
            if not self.degree:
                raise ValueError(
                    f'The "degree" parameter must be provided for a {POLY} kernel'
                )
            return partial(selected_kernel, self.degree)
        if self.kernel_name == POLY_STEP:
            if not self.num_labelled_points:
                raise ValueError(
                    f'The "num_labelled_points" parameter must be provided for a {POLY_STEP} kernel'
                )
            return partial(selected_kernel, self.num_labelled_points)
        return selected_kernel

    def _compute_kernel(self, tf_func, data):
        """ Step 1-4 of the proposed cluster kernel algorithm.
        (should probably be divided into multiple methods.)
        """

    def _linear_tf(self, lambda_):
        """Linear transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ : in-place modified array of eigenvalues"""

        return np.sort(lambda_)

    def _step_tf(self, lambda_, lambda_cut=None):
        """Step transfer function.
        Args :
            - lambda_ : array of eigenvalues
            - lambda_cut : thresholding value for the eigenvalues
                            if cutoff_type is absolute
        Output :
            - lambda_ :  modified array of eigenvalues"""

        lambda_ = np.sort(lambda_)
        if self.cutoff_type == 'n_relative':
            lambda_cut = lambda_[self.r]
        elif (self.cutoff_type == 'absolute') and (lambda_cut is None):
            raise ValueError('A threshold value for the eigenvalues has to be specified.')

        mask = lambda_>=lambda_cut
        return mask.astype('float64')

    def _linear_step_tf(self, lambda_, lambda_cut=None):
        """Linear-step transfer function.
        Args :
            - lambda_ : array of eigenvalues
            - lambda_cut : thresholding value for the eigenvalues
                            if cutoff_type is absolute
        Output :
            - lambda_ : modified array of eigenvalues"""

        lambda_ = np.sort(lambda_)
        if self.cutoff_type == 'n_relative':
            lambda_cut = lambda_[self.r]
        elif (self.cutoff_type == 'absolute') and (lambda_cut is None):
            raise ValueError('A threshold value for the eigenvalues has to be specified.')

        mask_over = lambda_>=lambda_cut
        mask_under = np.bitwise_xor(True, mask)
        lambda_[mask_under]=0
        return lambda_

    def _poly_tf(self, lambda_):
        """Polynomial transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ :  modified array of eigenvalues"""

        lambda_ = np.sort(lambda_)
        return np.power(lambda_, self.degree)

    def _poly_step_tf(self, lambda_):
        """Poly-step transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ : modified array of eigenvalues"""

        lambda_[:self.r] = np.power(lambda_[:self.r], self.p) # r first eigenvalues
        lambda_[self.r:] = np.power(lambda_[self.r:], self.q)

        return lambda_
