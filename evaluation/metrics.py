#
# This file includes the implemented metrics.
#
import numpy as np
import scipy as sc
import os
import multiprocessing as mp
from typing import Mapping


def get_mse(a: float, b: float, sd: bool = False) -> float:
    """Returns MSE across first axis of two arrays

    Args:
        a (float): Array one
        b (float): Array two
        sd (bool): Indicator whether to return standard deviation

    Returns:
        float: MSE
    """
    if sd == False:
        return np.mean(np.power(a - b, 2), axis=0)
    else:
        return np.mean(np.power(a - b, 2), axis=0), np.std(np.power(a - b, 2), axis=0)


def corr_func(h: float, model: str, r: float, s: float) -> float:
    """Calculates the correlation function (powexp, whitmat) or the variogram (brown) depending on the model parameters
    and distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter

    Returns:
        float: Returns value of correlation function / variogram
    """
    if model == "brown":
        res = np.power((h / r), s)
    elif model == "powexp":
        res = np.exp(-np.power((h / r), s))
    elif model == "whitmat":
        res = (
            np.power(2, (1 - s))
            / sc.special.gamma(s)
            * np.power((h / r), s)
            * sc.special.kv(s, (h / r))
        )
    else:
        res = None
    return res


def extremal_coefficient(h: float, model: str, r: float, s: float) -> float:
    """Calculates the extremal coefficient depending on the model, model parameters and distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter

    Returns:
        float: Returns value of the extremal coefficient
    """
    h = np.expand_dims(h, axis=1)
    if model == "brown":
        res = 2 * sc.special.ndtr(np.sqrt(corr_func(h, model, r, s)) / 2)
    else:
        res = 1 + np.sqrt((1 - corr_func(h, model, r, s)) / 2)
    return np.transpose(res)


def sample_extremal_coefficient(
    h: float, model: str, r: float, s: float, mean: bool = True
) -> float:
    """Calculates the extremal coefficient depending on the model, model parameters and distance h for a model with sample based output.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter
        mean (bool, optional): Boolean whether to calculate the mean or to return the complete array. Defaults to True.

    Returns:
        float: Returns value of the extremal coefficient
    """
    if mean:
        return np.transpose(np.mean(extremal_coefficient(h, model, r, s), axis=0))
    else:
        return extremal_coefficient(h, model, r, s)


def aggregate_direct_estimation(
    estimation: np.ndarray, type: str = "mean", q: float = None
) -> np.ndarray:
    """Aggregate the direct estimation of the extremal coefficient function for multiple samples.

    Args:
        estimation (np.ndarray): The array with discrete evaluation points of the extremal coefficient function.
        type (str, optional): Whether to calculate a quantile or a mean prediction. Defaults to "mean".
        q (float, optional): Quantile level. Defaults to None.

    Returns:
        np.ndarray: Predicted extremal coefficient function.
    """
    if type == "mean":
        return np.sort(np.mean(estimation, axis=2))
    elif type == "quantile":
        return np.sort(np.quantile(estimation, q=q, axis=2))


def error_function(
    h: float,
    model: str,
    true: Mapping[float, float],
    estimate: Mapping[float, float],
    method: str = "single",
    model2: str = None,
) -> float:
    """Generates the squared error between two extremal coefficient functions evaluated at the distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        true (Mapping[float, float]): Mapping with the two parameters r,s of the true model
        estimate (Mapping[float, float]): Mapping with the parameters r,s of the estimated model
        method (str): Whether to calculate the error based on a single estimation or multiple samples

    Returns:
        float: Squared error evaluated at h
    """
    model2 = model if model2 == None else model2
    r_true = true[:, 0]
    s_true = true[:, 1]
    r_est = estimate[:, 0:1]
    s_est = estimate[:, 1:2]
    if method == "single":
        r_est = np.squeeze(r_est)
        s_est = np.squeeze(s_est)
        error = np.power(
            extremal_coefficient(h, model, r_true, s_true)
            - extremal_coefficient(h, model2, r_est, s_est),
            2,
        )
    elif method == "sample":
        error = np.power(
            extremal_coefficient(h, model, r_true, s_true)
            - sample_extremal_coefficient(h, model2, r_est, s_est, True),
            2,
        )
    return error


def get_discrete_imse(
    model: str,
    dh: float = 0.1,
    true_parameters: float = None,
    estimate_parameters: float = None,
    estimate_function: np.array = None,
    method: str = "single",
    max_length: float = 42.5,
    sd: bool = False,
    model2: str = None,
) -> float:
    """_summary_

    Args:
        model (str): String describing the underlying model
        dh (float, optional): Step for distance h. Defaults to 0.1.
        true_parameters (float, optional): Array with the two parameters r,s of the true model. Defaults to None.
        estimate_parameters (float, optional): Array with the parameters r,s of the estimated model. Defaults to None.
        estimate_function (np.array, optional): Estimated extremal coefficient function. Defaults to None.
        method (str, optional): Whether to calculate the error based on a single estimation, multiple samples or direct estimation. Defaults to "single".
        max_length (float, optional): Used to compute the upper bound of integration. Defaults to 42.5.
        sd (bool, optional): Indicator whether to return standard deviation. Defaults to False.
        model2 (str, optional): String describing the estimated model. Defaults to None.

    Returns:
        float: Integrated mean squared error
    """

    model2 = model if model2 == None else model2
    h_support = np.arange(dh, max_length + dh, dh)

    if method == "direct":
        imse = np.power(
            extremal_coefficient(
                h_support, model, true_parameters[:, 0], true_parameters[:, 1]
            )
            - aggregate_direct_estimation(estimate_function, type="mean"),
            2,
        )
    else:
        imse = error_function(
            h_support, model, true_parameters, estimate_parameters, method, model2
        )
    imse = dh * np.sum(imse, axis=1)

    if sd == False:
        return np.mean(imse)
    else:
        return np.mean(imse), np.std(imse)


def get_interval_score(
    observations: float,
    alpha: float,
    q_left: float = None,
    q_right: float = None,
    full: bool = False,
    sd: bool = False,
) -> float:
    """Calculates the interval score for observations and a prediction interval at level alpha

    Args:
        observations (float): True values.
        alpha (float): Alpha value of quantile in (0,1)
        q_left (float, optional): Predicted lower quantile. Defaults to None.
        q_right (float, optional): Predicted upper quantile. Defaults to None.
        full (bool, optional): Whether to return all values, otherwise the mean is returned. Defaults to False.
        sd (bool, optional): Whether to return the standard deviation. Defaults to False.

    Returns:
        float: Interval score.
    """
    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    total = sharpness + calibration
    if full:
        return total
    if sd == False:
        return np.mean(total, axis=0)
    else:
        return np.mean(total, axis=0), np.std(total, axis=0)
    

def get_discrete_iis(
    model: str,
    dh: float = 0.1,
    true_parameters: float = None,
    estimate_parameters: float = None,
    estimate_function: np.array = None,
    method: str = "sample",
    max_length: float = 42.5,
    alpha: float = 0.05,
    sd: bool = False,
    model2: str = None,
) -> float:
    """ Calculate a disretized version of the integrated interval score over the distance h.

    Args:
        model (str): Model
        dh (float, optional): Step for distance h. Defaults to 0.1.
        true_parameters (float, optional): Array with the two parameters r,s of the true model. Defaults to None.
        estimate_parameters (float, optional): Array with the parameters r,s of the estimated model. Defaults to None.
        estimate_function (np.array, optional): Estimated extremal coefficient function. Defaults to None.
        method (str, optional): Whether to calculate the error based on multiple samples or direct estimation. Defaults to "sample".
        max_length (float, optional): Used to compute the upper bound of integration. Defaults to 42.5.
        alpha (float, optional): Alpha value of quantile in (0,1). Defaults to 0.05.
        sd (bool, optional): Whether to return the standard deviation. Defaults to False.
        model2 (str, optional): String describing the estimated model. Defaults to None.

    Returns:
        float: Integrated interval score
    """
    
    # Check for second model
    model2 = model if model2 == None else model2
    h_support = np.arange(dh, max_length + dh, dh)

    # Calculate true extremal coefficient functions
    true_theta = extremal_coefficient(h_support, model, true_parameters[:, 0], true_parameters[:, 1])

    if method == "direct":
        left = aggregate_direct_estimation(estimate_function, type = "quantile", q = alpha/2)
        right = aggregate_direct_estimation(estimate_function, type = "quantile", q = 1- alpha/2)
        interval_score = get_interval_score(true_theta, alpha = alpha, q_left = left, q_right = right, full = True)
        iis = interval_score.sum(axis = 1) * dh
    else:
        theta_aggregated = extremal_coefficient(h_support, model2, estimate_parameters[:,0:1], estimate_parameters[:,1:2])
        left = np.transpose(np.quantile(theta_aggregated, 0.025, axis=0))
        right = np.transpose(np.quantile(theta_aggregated, 0.975, axis=0))
        interval_score = get_interval_score(true_theta, alpha = alpha, q_left = left, q_right = right, full = True)
        iis = interval_score.sum(axis = 1) * dh

        
    if sd == False:
        return np.mean(iis)
    else:
        return np.mean(iis), np.std(iis)    


def get_energy_score(y_true: float, y_pred: float, sd: bool = False) -> float:
    """Compute mean energy score from samples of the predictive distribution.

    Args:
        y_true (float): True values, shape = (n_samples, n_dim).
        y_pred (float): Samples from predictive distribution, shape = (n_samples, n_dim, n_gen_samples).
        sd (bool, optional): _description_. Defaults to False.

    Returns:
        float: Mean energy score
    """

    N = y_true.shape[0]
    M = y_pred.shape[2]

    es_12 = np.zeros(y_true.shape[0])
    es_22 = np.zeros(y_true.shape[0])

    for i in range(N):
        es_12[i] = np.sum(
            np.sqrt(np.sum(np.square((y_true[[i], :].T - y_pred[i, :, :])), axis=0))
        )
        es_22[i] = np.sum(
            np.sqrt(
                np.sum(
                    np.square(
                        np.expand_dims(y_pred[i, :, :], axis=2)
                        - np.expand_dims(y_pred[i, :, :], axis=1)
                    ),
                    axis=0,
                )
            )
        )

    scores = es_12 / M - 0.5 * 1 / (M * M) * es_22
    if sd == False:
        return np.mean(scores)
    else:
        return np.mean(scores), np.std(scores)
    

def get_functional_energy_score(
    model: str,
    dh: float = 0.1,
    true_parameters: float = None,
    estimate_parameters: float = None,
    estimate_function: np.array = None,
    method: str = "sample",
    max_length: float = 42.5,
    sd: bool = False,
    model2: str = None,
) -> float:
    """Compute mean energy score from support points of the extremal coefficient function.

    Args:
        modle (str): Model
        dh (float, optional): Step for distance h. Defaults to 0.1.
        true_parameters (float, optional): Array with the two parameters r,s of the true model. Defaults to None.
        estimate_parameters (float, optional): Array with the parameters r,s of the estimated model. Defaults to None.
        estimate_function (np.array, optional): Estimated extremal coefficient function. Defaults to None.
        method (str, optional): Whether to calculate the error based on multiple samples or direct estimation. Defaults to "sample".
        max_length (float, optional): Used to compute the upper bound of integration. Defaults to 42.5.
        alpha (float, optional): Alpha value of quantile in (0,1). Defaults to 0.05.
        sd (bool, optional): Whether to return the standard deviation. Defaults to False.
        model2 (str, optional): String describing the estimated model. Defaults to None.
    Returns:
        float: Mean energy score
    """

    model2 = model if model2 == None else model2    
    h_support = np.arange(dh, max_length + dh, dh)
    # Calculate true extremal coefficient functions
    true_theta = extremal_coefficient(h_support, model, true_parameters[:, 0], true_parameters[:, 1])
    
    if method == "direct":
        es,std = get_energy_score(y_true = true_theta, y_pred = estimate_function, sd = True)
    else:
        theta_aggregated = extremal_coefficient(h_support, model2, estimate_parameters[:,0:1], estimate_parameters[:,1:2])
        es,std = get_energy_score(y_true = true_theta, y_pred = theta_aggregated, sd = True)

    if sd == False:
        return es
    else:
        return es,std


def bivariate_cdf(
    z1: float, z2: float, h: float, model: str, r: float, s: float
) -> float:
    """Returns the bivariate CDF of a max stable-process, specified by the parameters, at the points z1,z2

    Args:
        z1 (float): Evaluation point
        z2 (float): Evaluation point
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter

    Returns:
        float: CDF of max-stable process
    """
    rho = corr_func(h, model, r, s)
    if model == "brown":
        a = np.sqrt(2 * rho)
        Phi = lambda z1, z2: sc.special.ndtr(a / 2 + (1 / a) * (np.log(z2 / z1)))
        V = (1 / z1) * Phi(z1, z2) + (1 / z2) * Phi(z2, z1)
    else:
        V = (
            0.5
            * (1 / z1 + 1 / z2)
            * (1 + np.sqrt(1 - (2 * (1 + rho) * z1 * z2) / (np.power(z1 + z2, 2))))
        )
    cdf = np.exp(-V)
    return cdf


def bivariate_density(
    z1: float, z2: float, h: float, model: str, r: float, s: float
) -> float:
    """Returns the bivariate density of a max stable-process, specified by the parameters, at the points z1,z2

    Args:
        z1 (float): Evaluation point
        z2 (float): Evaluation point
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter

    Returns:
        float: Density of max-stable process
    """
    rho = corr_func(h, model, r, s)
    # Brown-Resnick
    if model == "brown":
        rho = np.sqrt(2 * rho)
        # Define helping terms
        phi = (
            lambda z1, z2, a: 1
            / (np.sqrt(2 * np.pi))
            * np.exp(-0.5 * np.power(a / 2 + (1 / a) * (np.log(z2 / z1)), 2))
        )
        Phi = lambda z1, z2, a: sc.special.ndtr(a / 2 + (1 / a) * (np.log(z2 / z1)))

        V = lambda z1, z2, a: (1 / z1) * Phi(z1, z2, a) + (1 / z2) * Phi(z2, z1, a)
        V1 = (
            lambda z1, z2, a: (-1 / np.power(z1, 2)) * Phi(z1, z2, a)
            - (1 / (a * np.power(z1, 2))) * phi(z1, z2, a)
            + (1 / (a * z1 * z2)) * phi(z2, z1, a)
        )
        V12 = (
            lambda z1, z2, a: (-1 / (a * z2 * np.power(z1, 2))) * phi(z1, z2, a)
            + (1 / (np.power(a, 2) * z2 * np.power(z1, 2)))
            * phi(z1, z2, a)
            * (a / 2 + 1 / a * np.log(z2 / z1))
            - (1 / (np.power(z2, 2) * a * z1)) * phi(z2, z1, a)
            + (1 / (z1 * np.power(z2, 2) * np.power(a, 2)))
            * phi(z2, z1, a)
            * (a / 2 + 1 / a * np.log(z1 / z2))
        )

    # Schlather
    else:
        V = (
            lambda z1, z2, rho: 0.5
            * (1 / z1 + 1 / z2)
            * (
                1
                + 1
                / (z1 + z2)
                * np.sqrt(np.power(z1, 2) - 2 * z1 * z2 * rho + np.power(z2, 2))
            )
        )
        V1 = lambda z1, z2, rho: -1 / (2 * np.power(z1, 2)) + 1 / 2 * (
            rho / z1 - z2 / np.power(z1, 2)
        ) * np.power(np.power(z1, 2) - 2 * z1 * z2 * rho + np.power(z2, 2), -0.5)
        V12 = (
            lambda z1, z2, rho: -1
            / 2
            * (1 - np.power(rho, 2))
            * (np.power(np.power(z1, 2) - 2 * z1 * z2 * rho + np.power(z2, 2), -3 / 2))
        )

    density = np.exp(-V(z1, z2, rho)) * (
        V1(z1, z2, rho) * V1(z2, z1, rho) - V12(z1, z2, rho)
    )
    return density
