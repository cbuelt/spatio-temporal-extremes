# This file includes configurations for training the model

config = dict()

# Define types of neural networks to train and evaluate
config["network_types"] = ["normal", "energy", "energy_theta"]
# Number of epochs
config["epochs"] = 100
# Batch size
config["batch_size"] = 100
# Test size of dataset
config["test_size"] = 250 # 250 for simulations, 9 for application
# Support points of extremal coefficient function
config["dh"] = 0.1
# Maximum distance across observed processes
config["max_length"] = 42.5
# Models per directory
config["models_per_dir"] = {
    "normal": ["brown", "powexp"],
    "outside_model": ["brown", "whitmat"],
    "outside_parameters": ["brown"],
    "all_models_small": ["brown", "powexp"],
    "all_models_large": ["brown", "powexp"],
}
