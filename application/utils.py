import numpy as np
import cartopy.feature as cfeature

def get_gev_params():
    """ Returns a dictionary with the GEV parameters.
    """
    gev_params = {
    "loc1": 64.799623725,
    "loc_lat": -0.999680643,
    "loc_lon": 0.014852714,
    "loc_time": 0.001105532,
    "scale": 7.004518221,
    "shape": 0.105249698,
}
    return gev_params

def gev2frech(data, year):
    """ Converts data from GEV to unit Fréchet.

    Args:
        data (_type_): Observed spatial data.
        year (_type_): The year as a covariate.

    Returns:
        _type_: Transformed spatial data.
    """
    # Get parameters
    gev_params = get_gev_params()
    # Define lat lon grid
    lons = np.linspace(6.38, 7.48, 30)
    lats = np.linspace(50.27, 50.97, 30)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Define year
    year_emb = year - 1931 + 1

    # Calculate parameters
    mu = (
        gev_params["loc1"]
        + lat2d * gev_params["loc_lat"]
        + lon2d * gev_params["loc_lon"]
        + year_emb * gev_params["loc_time"]
    )
    sigma = gev_params["scale"]
    gamma = gev_params["shape"]

    # Transformation
    result = np.power(1 + gamma * (data - mu) / sigma, (1 / gamma))
    return result

def frech2gev(data, year):
    """ Converts data from unit Fréchet to GEV.

    Args:
        data (_type_): Observed spatial data.
        year (_type_): The year as a covariate.

    Returns:
        _type_: Transformed spatial data.
    """
    # Get parameters
    gev_params = get_gev_params()
    # Define lat lon grid
    lons = np.linspace(6.38, 7.48, 30)
    lats = np.linspace(50.27, 50.97, 30)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Define year
    year_emb = year - 1931 + 1

    # Calculate parameters
    mu = (
        gev_params["loc1"]
        + lat2d * gev_params["loc_lat"]
        + lon2d * gev_params["loc_lon"]
        + year_emb * gev_params["loc_time"]
    )
    sigma = gev_params["scale"]
    gamma = gev_params["shape"]

    # Transformation
    result = mu + sigma * (np.power(data, gamma)-1)/gamma
    return result

def plot_cities(axs, river = False):
    """ Function for adding cities to a plot.
    """
    axs.plot(7.10066, 50.735851, ".", color = "black", markersize=15)
    axs.text(7.12, 50.735851, 'Bonn', fontsize = 20)

    axs.plot(6.959974, 50.938361, ".", color = "black", markersize=15)
    axs.text(6.985, 50.938361, 'Köln', fontsize = 20)

    axs.plot(6.486739, 50.799552, ".", color = "black", markersize=15)
    axs.text(6.50, 50.799552, 'Düren', fontsize = 20)

    axs.plot(6.988617, 50.513673, ".", color = "black", markersize=15)
    axs.text(7, 50.513673, 'Altenahr', fontsize = 20)

    if river:
        axs.add_feature(cfeature.RIVERS, linewidth = 6, alpha = 0.8)


def frechet_margins(x):
    """ Function for Fréchet margins.
    """
    return np.exp(- 1/x)

def f_madogram(z1, z2):
    """ Function for f-madogram transform.
    """
    n = len(z1)
    return 1/(2*n)*np.sum((np.abs(frechet_margins(z1) - frechet_margins(z2))))
