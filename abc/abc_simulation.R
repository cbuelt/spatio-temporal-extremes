#
# This script can be used to simulate samples from the ABC algorithm
#
current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Load functions from other file
source("abc_functions.R")
#Get nodes
no_cores <- detectCores() - 6


#Define grid, directory and model
dir <- "normal"
model <- "brown"
# Define grid
length <- 30
x <- seq(0, length, length = length)
grid <- expand.grid(x, x)
grid <- array(unlist(grid), dim = c(length ** 2, 2))

# Define smaller grid for downsampling
downsample_size <- 5
x_small <- seq(0, 30, length = downsample_size)
grid_small <- expand.grid(x_small, x_small)
grid <- array(unlist(grid_small), dim = c(downsample_size ** 2, 2))

#Load corresponding data
load(paste0("../data/", dir, "/data/", model, "_test_data.RData"))
load(paste0("../data/", dir, "/data/", model, "_test_params.RData"))
# Number of test samples
n_test <- 250

#Measure time
start_time <- Sys.time()

#Run simulations
index <- seq(1, n_test, 1)
result <-
  sapply(
    X = index,
    FUN = abc_wrapper,
    simplify = "array",
    gid = grid,
    model = model,
    n_sim = 50000,
    n_cores = no_cores,
    interp = "interpolate",
    n_sim_each = 25
  )
#Save results
save(result,
     file = paste0("../data/", dir, "/results/", model, "_abc_results.RData"))
# Alternatively save results as netCDF
ArrayToNc(list(results = result), paste0("../data/", dir, "/results/", model, "_abc_results.nc"))
print(dim(result))
print(Sys.time() - start_time)
