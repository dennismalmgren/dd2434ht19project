"""
Module containing the implementation of various cluster kernels.
"""
from functools import partial
import gin
from utils import LINEAR, LINEAR_STEP, STEP, POLY, POLY_STEP


@gin.configurable
class ClusterKernel:
    def __init__(self,
                 kernel_name=LINEAR,
                 degree=None,
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
        """Linear transfer function."""

    def _step_tf(self, lambda_):
        """Step transfer function."""

    def _linear_step_tf(self, lambda_):
        """Linear-step transfer function."""

    def _poly_tf(self, lambda_, degree):
        """Polynomial transfer function."""

    def _poly_step_tf(self, lambda_, num_labelled_points):
        """Poly-step transfer function."""
