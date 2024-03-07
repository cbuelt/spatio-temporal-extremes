# Set path
current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
# Source functions
source("pl_functions.R")
#Get nodes
no_cores <- detectCores() - 2


# Define model and directory
model <- "brown"
dir <- "normal"
load(paste0("../data/", dir, "/data/", model, "_test_data.RData"))
load(paste0("../data/", dir, "/data/", model, "_test_params.RData"))
n_param <- dim(test_params)[1]

# Define grid used in data
length <- 30
x <- seq(0, length, length = length)
grid <- expand.grid(x, x)
grid <- array(unlist(grid), dim = c(length ** 2, 2))




# Code used for simulating data for the application scenario

# Load application data
model <- "whitmat"
library(ncdf4)
dir <- "application"
nc_data <- nc_open(paste0("../data/", dir, "/data/", "test_data.nc"))
test_data <-  array(ncvar_get(nc_data, "pr"), dim = c(900, 9))
nc_close(nc_data)
n_param <- dim(test_data)[2]
length <- 30
grid_nc <- nc_open(paste0("../data/", dir, "/data/", "grid.nc"))
grid <-  ncvar_get(grid_nc, "grid")
nc_close(grid_nc)



# Get weights
weights <- get_weights(grid, length, cutoff = 5)

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl, c('run_start_values', "model", "length"))
clusterEvalQ(cl, library(SpatialExtremes))

# Run estimations
res <-
  parSapply(cl, seq(1, n_param), test_data, grid, weights, FUN = apply_mle)
stopCluster(cl)
results <- t(res)
save(results,
     file = paste0(
       "../data/",
       dir,
       "/results/",
       model,
       "_pl.RData"
     ))


