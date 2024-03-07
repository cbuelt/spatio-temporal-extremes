#
# This file can be used to calculate the metrics and save them as a pandas dataframe.
#

from metrics import *
import pyreadr
import xarray as xr
import pandas as pd

def fill_metrics(results, model, true_parameters, pl, abc, cnn, cnn_e, cnn_direct, alpha = 0.05, model2 = None):
    """ Given a results table, this method caluclates the corresponding metrics.

    Args:
        results (_type_): An empty dataframe in a specified format.
        model (_type_): The max-stable model.
        true_parameters (_type_): Array of the true parameters.
        pl (_type_): Array of the estimated parameters (PL).
        abc (_type_): Array of the estimated parameters (ABC).
        cnn (_type_): Array of the estimated parameters (CNN).
        cnn_e (_type_): Array of the estimated parameters (CNN_E).
        alpha (float, optional): Alpha quantile for the interval score. Defaults to 0.05.
        model2 (_type_, optional): Second model, if comparing two different ones. Defaults to None.

    Returns:
        _type_: A filled pandas dataframe.
    """
    #Get mean prediction
    abc_mean = np.mean(abc, axis = 2)
    cnn_es_mean = np.mean(cnn_e, axis = 2)
    # PL
    mse = get_mse(true_parameters, pl, sd = True)
    imse = get_discrete_imse(model, true_parameters = true_parameters, estimate_parameters = pl, sd = True, model2 = model2)
    results.loc[(model, "MSE_r"), "PL"] = f"{mse[0][0]:.2f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "PL"] = f"{mse[0][1]:.2f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "PL"] = f"{imse[0]:.2f} ({imse[1]:.2f})"

    # CNN 
    mse = get_mse(true_parameters, cnn, sd = True)
    imse = get_discrete_imse(model, true_parameters = true_parameters, estimate_parameters = cnn, sd = True, model2 = model2)
    results.loc[(model, "MSE_r"), "CNN"] = f"{mse[0][0]:.2f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "CNN"] = f"{mse[0][1]:.2f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "CNN"] = f"{imse[0]:.2f} ({imse[1]:.2f})"

    # ABC
    mse = get_mse(true_parameters, abc_mean, sd = True)
    imse = get_discrete_imse(model, true_parameters = true_parameters, estimate_parameters = abc, method = "sample", sd = True, model2 = model2)
    results.loc[(model, "MSE_r"), "ABC"] = f"{mse[0][0]:.2f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "ABC"] = f"{mse[0][1]:.2f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "ABC"] = f"{imse[0]:.2f} ({imse[1]:.2f})"

    quantiles = np.quantile(abc, [alpha/2,1-(alpha/2)], axis = 2)
    iscore = get_interval_score(true_parameters, alpha = alpha, q_left = quantiles[0], q_right = quantiles[1], sd = True)
    iis = get_discrete_iis(model, true_parameters = true_parameters, estimate_parameters = abc, alpha = alpha, sd = True, model2 = model2)
    es = get_energy_score(true_parameters,abc, sd = True)
    es_func = get_functional_energy_score(model = model, true_parameters = true_parameters, estimate_parameters = abc, sd = True, model2 = model2)
    results.loc[(model, "IS_r",), "ABC"] = f"{iscore[0][0]:.2f} ({iscore[1][0]:.2f})"
    results.loc[(model, "IS_s",), "ABC"] = f"{iscore[0][1]:.2f} ({iscore[1][1]:.2f})"
    results.loc[(model, ["IIS"]), "ABC"] = f"{iis[0]:.2f} ({iis[1]:.2f})"
    results.loc[(model, ["ES"]), "ABC"] = f"{es[0]:.2f} ({es[1]:.2f})"
    results.loc[(model, ["ES2"]), "ABC"] = f"{es_func[0]:.2f} ({es_func[1]:.2f})"

    # CNN ES
    mse = get_mse(true_parameters, cnn_es_mean, sd = True)
    imse = get_discrete_imse(model, true_parameters = true_parameters, estimate_parameters = cnn_e, method = "sample", sd = True, model2 = model2)
    results.loc[(model, "MSE_r"), "CNN_ES"] = f"{mse[0][0]:.2f} ({mse[1][0]:.2f})"
    results.loc[(model, "MSE_s"), "CNN_ES"] = f"{mse[0][1]:.2f} ({mse[1][1]:.2f})"
    results.loc[(model, "MSE_ext"), "CNN_ES"] = f"{imse[0]:.2f} ({imse[1]:.2f})"

    quantiles = np.quantile(cnn_e, [alpha/2,1-(alpha/2)], axis = 2)
    iscore = get_interval_score(true_parameters, alpha = alpha, q_left = quantiles[0], q_right = quantiles[1], sd = True)
    iis = get_discrete_iis(model, true_parameters = true_parameters, estimate_parameters = cnn_e, alpha = alpha, sd = True, model2 = model2)
    es = get_energy_score(true_parameters, cnn_e, sd = True)
    es_func = get_functional_energy_score(model = model, true_parameters = true_parameters, estimate_parameters = cnn_e, sd = True, model2 = model2)
    results.loc[(model, "IS_r",), "CNN_ES"] = f"{iscore[0][0]:.2f} ({iscore[1][0]:.2f})"
    results.loc[(model, "IS_s",), "CNN_ES"] = f"{iscore[0][1]:.2f} ({iscore[1][1]:.2f})"
    results.loc[(model, ["IIS"]), "CNN_ES"] = f"{iis[0]:.2f} ({iis[1]:.2f})"
    results.loc[(model, ["ES"]), "CNN_ES"] = f"{es[0]:.2f} ({es[1]:.2f})"
    results.loc[(model, ["ES2"]), "CNN_ES"] = f"{es_func[0]:.2f} ({es_func[1]:.2f})"

    # CNN direct
    imse = get_discrete_imse(model, true_parameters = true_parameters, estimate_function = cnn_direct, method = "direct", sd = True, model2 = model2)
    iis = get_discrete_iis(model, true_parameters = true_parameters, estimate_function = cnn_direct, method = "direct", alpha = alpha, sd = True, model2 = model2)
    es_func = get_functional_energy_score(model = model, true_parameters = true_parameters, estimate_function = cnn_direct, method = "direct", sd = True, model2 = model2)
    results.loc[(model, "MSE_ext"), "CNN_ES_direct"] = f"{imse[0]:.2f} ({imse[1]:.2f})"
    results.loc[(model, ["IIS"]), "CNN_ES_direct"] = f"{iis[0]:.2f} ({iis[1]:.2f})"
    results.loc[(model, ["ES2"]), "CNN_ES_direct"] = f"{es_func[0]:.2f} ({es_func[1]:.2f})"

    return results


def load_predictions(data_path, results_path, model, transform = False):
    """ This function loads the different predictions.

    Args:
        data_path (_type_): The path for loading the data.
        results_path (_type_): The path for the results.
        model (_type_): The max-stable model.

    Returns:
        _type_: The true and estimated parameters.
    """
    # Load true parameters
    true_parameters = np.load(data_path+model+"_test_params.npy")[0:n_test]
    # Load PL
    pl = pyreadr.read_r(results_path+model+"_pl.RData")["results"].to_numpy()[0:n_test,0:2]
    # Load ABC
    abc = xr.open_dataset(results_path + model + "_abc_results.nc").results.data[0:n_test,0:2]
    # Load normal network
    cnn = np.load(results_path+model+"_cnn.npy")[0:n_test]
    # Load energy network
    cnn_e = np.load(results_path+model+"_cnn_es.npy")[0:n_test]
    # Load direct estimation
    cnn_direct = np.load(results_path+model+"_cnn_es_theta.npy")[0:n_test]

    # Transform for Smith model
    if transform:
        true_parameters[:,0] = np.sqrt(2 * true_parameters[:,0])

    return true_parameters, pl, abc, cnn, cnn_e, cnn_direct

def get_results_table(dir, models, model2 = None, save = True, transform = False):
    """ This function calculates the results table for a given model and directory.

    Args:
        dir (_type_): Directoy of the data.
        models (_type_): A list of max-stable models to evaluate
        model2 (_type_, optional): A second model if comparing two different. Defaults to None.
        save (bool, optional): Whether to save the results. Defaults to True.

    Returns:
        _type_: Pandas dataframe with results.
    """
    # Set data paths
    data_path = f'data/{dir}/data/'
    results_path = f'data/{dir}/results/'
    # Set filename
    filename = models[0] + "_results_table.pkl" if dir == "outside_model" else "results_table.pkl"

    # Prepare dataframe
    metrics = ["MSE_r", "MSE_s", "MSE_ext", "IS_r", "IS_s", "IIS", "ES", "ES2"]
    results = pd.DataFrame("-", index = pd.MultiIndex.from_product([models, metrics]), columns = ["PL", "CNN", "ABC", "CNN_ES", "CNN_ES_direct"])
    
    for model in models:
        true_parameters, pl, abc, cnn, cnn_es, cnn_direct = load_predictions(data_path, results_path, model, transform = transform)
        results = fill_metrics(results, model, true_parameters, pl, abc, cnn, cnn_es, cnn_direct, model2 = model2)

    if save:
        results.to_pickle(results_path + filename)
        print(f"Successfully saved table at {results_path + filename}")
    else:
        return results


if __name__ == "__main__":
    n_test = 250

    # Normal predictions
    exp = "normal"
    get_results_table(exp, models = ["brown", "powexp"])    

    # Outside parameters
    exp = "outside_parameters"
    get_results_table(exp, models = ["brown"])

    # Outside model - Whitmat
    exp = "outside_model"
    get_results_table(exp, models = ["whitmat"], model2 = "powexp")

    # Outside model - Smith
    get_results_table(exp, models = ["brown"], transform = True)

    # Aggregated data
   # exp = "all_models_small"
   # get_results_table(exp, models = ["brown", "powexp"])

   # exp = "all_models_large"
  #  get_results_table(exp, models = ["brown", "powexp"])
    






    