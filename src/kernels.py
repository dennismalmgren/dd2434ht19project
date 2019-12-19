"""
Module containing the implementation of various cluster kernels.
"""
from functools import partial
import numpy as np
import gin
from constants import *

@gin.configurable
class ClusterKernel:
    def __init__(self,
                 kernel_name=LINEAR,
                 degree=None,
                 sigma=.55, #used by RBF kernel
                 cutoff_type=CUTOFF_RELATIVE, #either n_relative or absolute
                 lambda_cut=None,
                 r=10, #This defines cutoff for step, linear-step and poly-step.
                 p=2, #Power for poly-step under or equal to cutoff
                 q=2 #Power for poly-step over cutoff
                 ):
        """
        Generates a ClusterKernel of the type kernel_name.
        To compute the kernel for data X, call obj.kernel(X)
        where obj is the name of the ClusterKernel instance.

        Args:
            - degree: required if 'poly' kernel is selected
            - num_labelled_points: required if 'poly-step' kernel is selected
        """
        self.sigma = sigma
        self.cutoff_type = cutoff_type
        self.lambda_cut = lambda_cut
        self.r = r
        self.p = p
        self.q = q
        self.kernel_name = kernel_name
        self.degree = degree
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
        if self.cutoff_type == CUTOFF_ABSOLUTE and self.lambda_cut is None:
            raise ValueError(
                'The "lambda_cut" parameter must be provided for step kernels'
                'if the "cutoff_type" is absolute"'
            )
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
        # sort eigenvalues in descending order
        # idx = eig_vals.argsort()[::-1]
        # eig_vals = eig_vals[idx]
        # eig_vecs = eig_vecs[:, idx]

        # Step 3
        lambda_eig_vals = tf_func(eig_vals)
        lambda_L = eig_vecs @ np.diag(lambda_eig_vals) @ eig_vecs.T

        # Step 4
        diag_lambda_L = np.diagonal(lambda_L)
        lambda_D = np.diag(1/diag_lambda_L)
        lambda_K = lambda_D**(.5) @ lambda_L @ lambda_D**(.5)
        def kernel_fun(x, y):
            #Just checking x..
            if not isinstance(x, (list, tuple, np.ndarray)):
                x = int(x)
                y = int(y)
                return lambda_K[x][y]
            else:
                return lambda_K[x, :][:, y] #wow easy

        return kernel_fun

    def _rbf_kernel(self, data, sigma):
        """RBF kernel used by step 1 of the cluster kernel."""
        # assumes data.shape = (N, D)
        data_norm = np.linalg.norm(data, axis=1)**2
        pairwise_distance = (data_norm.reshape(-1, 1)
                             - 2*np.dot(data, data.T)
                             + data_norm.reshape(1, -1))
        return np.exp(-pairwise_distance/(2*sigma**2))

    def _linear_tf(self, lambda_):
        """Linear transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ : modified array of eigenvalues"""

        return lambda_

    def get_larger_eigenvalues(self, lambda_):
        """Given an array of eigenvalues, return the indices corresponding to
        the eigenvalues above and below the threshold.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - mask_over :  indexes of the array corresponding to the higher eigenvalues
            - mask_under :  indexes of the array corresponding to the lower eigenvalues"""

        if self.cutoff_type == CUTOFF_RELATIVE:
            self.lambda_cut = sorted(lambda_, reverse=True)[self.r]

        mask_over = lambda_ >= self.lambda_cut
        mask_under = lambda_ < self.lambda_cut

        return mask_over, mask_under


    def _step_tf(self, lambda_):
        """Step transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ :  modified array of eigenvalues"""

        mask_over, mask_under = self.get_larger_eigenvalues(lambda_)
        lambda_[mask_over] = 1.
        lambda_[mask_under] = 0.

        return lambda_

    def _linear_step_tf(self, lambda_):
        """Linear-step transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ : modified array of eigenvalues"""

        _, mask_under = self.get_larger_eigenvalues(lambda_)
        lambda_[mask_under] = 0.
        return lambda_

    def _poly_tf(self, lambda_):
        """Polynomial transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ :  modified array of eigenvalues"""

        return np.power(lambda_, self.degree)

    def _poly_step_tf(self, lambda_):
        """Poly-step transfer function.
        Args :
            - lambda_ : array of eigenvalues
        Output :
            - lambda_ : modified array of eigenvalues"""

        indexes_over, indexes_under = self.get_larger_eigenvalues(lambda_)

        lambda_[indexes_under] = np.power(lambda_[indexes_under], self.q)
        lambda_[indexes_over] = np.power(lambda_[indexes_over], self.p)

        return lambda_
