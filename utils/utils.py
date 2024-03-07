#
# This file includes general utility functions.
#

import numpy as np
import pyreadr


def transform_parameters(params):
    """
    Transform parameters.
    First parameter is log transformed.
    Second parameter is transformed to [0,1] range.

    Args:
        params : Input parameters

    Returns:
        result: Transformed parameters
    """
    result = np.zeros(shape=params.shape).astype("float32")
    result[0] = np.log(params[0])
    result[1] = params[1] / 2
    return result


def retransform_parameters(params):
    """
    Retransform parameters.
    First parameter is transformed with exponential.
    Second parameter is transformed to [0,2] range.

    Args:
        params : Input parameters

    Returns:
        result: Transformed parameters
    """
    result = np.zeros(shape=params.shape)
    result[:, 0] = np.exp(params[:, 0])
    result[:, 1] = params[:, 1] * 2
    return result


def generate_support_points(dh=0.1, max_length=42.5) -> np.ndarray:
    """Generate support points for evaluating a funciton.

    Args:
        dh (float, optional): Defaults to 0.1.
        max_length (float, optional): Defaults to 42.5.

    Returns:
        _type_: Vector of support points
    """
    h_support = np.arange(dh, max_length + dh, dh)
    return h_support
