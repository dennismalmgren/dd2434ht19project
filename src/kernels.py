"""
Module containing the implementation of various cluster kernels.
"""
from functools import partial
import numpy as np
import gin
from utils import LINEAR, LINEAR_STEP, STEP, POLY, POLY_STEP


@gin.configurable
class ClusterKernel:
    def __init__(self,
                 kernel_name=LINEAR,
                 degree=None,
                 sigma=.55, #used by RBF kernel
                 cutoff_type = 'n_relative', #either n_relative or absolute
                 r = 10, #This defines cutoff for step and poly-step.
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
        self.sigma = sigma
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
        """ Step 1-4 of the proposed cluster kernel algorithm."""

        # Step 1
        matrix_K = self._rbf_kernel(data, self.sigma)
        diag_K = np.sum(matrix_K, axis=1)
        matrix_D = np.diag(diag_K)

        # Step 2
        diag_K_pow_neg_half = diag_K**(-.5)
        matrix_D_pow_neg_half = np.diag(diag_K_pow_neg_half)
        matrix_L = matrix_D_pow_neg_half @ matrix_K @ matrix_D_pow_neg_half
        eig_vals, eig_vecs = np.linalg.eig(matrix_L)

        # Step 3
        lambda_eig_vals = tf_func(eig_vals)
        lambda_L = eig_vecs @ np.diag(lambda_eig_vals) @ eig_vecs.T

        # Step 4
        diag_lambda_L = np.diagonal(lambda_L)
        lambda_D = np.diag(1/diag_lambda_L)
        lambda_K = lambda_D**(.5) @ lambda_L @ lambda_D**(.5)
        return lambda_K

    def _rbf_kernel(self, data, sigma):
        """RBF kernel used by step 1 of the cluster kernel."""
        # assumes data.shape = (N, D)
        data_norm = np.linalg.norm(data, axis=1)**2
        pairwise_distance = (data_norm.reshape(-1, 1)
                             - 2*np.dot(data, data.T)
                             + data_norm.reshape(1, -1))
        return np.exp(-pairwise_distance/(2*sigma**2))

    def _linear_tf(self, lambda_):
        """Linear transfer function."""
        return lambda_

    def _step_tf(self, lambda_):
        """Step transfer function."""

    def _linear_step_tf(self, lambda_):
        """Linear-step transfer function."""

    def _poly_tf(self, lambda_, degree):
        """Polynomial transfer function."""

    def _poly_step_tf(self, lambda_, num_labelled_points):
        """Poly-step transfer function."""
