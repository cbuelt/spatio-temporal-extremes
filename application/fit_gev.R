#
# This file is used to create the parameters for the application scenario
#
library(SpatialExtremes)
library(easyNCDF)
library(geosphere)
library(proj4)
library(ncdf4)
library(parallel)
current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))




# Define grid
lons <- seq(6.38,7.48,1.1/29)
lats <- seq(50.27,50.97,0.7/29)
lat_lon_grid <- array(unlist(expand.grid(lons,lats)), dim = c(30**2,2))
colnames(lat_lon_grid) <- c("lon", "lat")

# Transform grid into kilometer length instead of coordinates
grid <- ptransform(lat_lon_grid/180*pi, '+proj=longlat', '+proj=eqc') 
grid[,1] <- grid[,1] - grid[1,1]
grid[,2] <- grid[,2] - grid[1,2]
# Transform grid to units of 3.5km
grid <- grid[,1:2]/3400
sqrt(sum(grid[900,]**2))
ArrayToNc(list(grid = grid), "../data/application/data/grid.nc")
save(grid, file = paste0("../data/application/data/grid.RData"))

#
# GEV parameter estimation
#

data_raw <- nc_open("../data/application/data/1931_2020_month_max.nc")
data <- ncvar_get(data_raw, "pr")
n_obs <- dim(data)[3]
data_vec <- as.vector(data)
data_mat <- t(matrix(data_vec, nrow = 900, ncol = n_obs))

# Estimate spatial GEV
loc <- y ~ 1 + lat + lon
scale <- y ~ 1
shape <- y ~ 1
temp_loc <- y ~ year
temp_cov <- cbind(rep(seq(1,90), each = 3))
colnames(temp_cov) <- c("year")
gev_fit <- fitspatgev(data_mat, lat_lon_grid, loc, scale, shape, temp.cov = temp_cov, temp.form.loc = temp_loc)

# Print estimates
round(gev_fit$fitted.values,4)
round(gev_fit$std.err,4)


#
### Simulate data from estimated models
#
no_cores <- detectCores() - 2

simulate <- function(params, grid){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  data <- rmaxstab(1, coord = grid, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth)
  return(data)
}

#Load estimates and simulate 10 samples
parameter_estimates <- nc_open("../data/application/data/parameter_estimates.nc")
n_samples <- 10


for (month in c("June", "July", "August")){
  params <- ncvar_get(parameter_estimates, month)
  params <- array(rep(params, each = n_samples), dim = c(n_samples, 2))
  colnames(params) <- c("range", "smooth")
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('simulate'))
  clusterEvalQ(cl, library(SpatialExtremes))
  simulated_field <- parApply(cl, params, grid, MARGIN = 1, FUN = simulate)
  stopCluster(cl)
  
  #Save data
  save(simulated_field, file = paste0("../data/application/data/simulations_", month, "2022.RData"))
}


