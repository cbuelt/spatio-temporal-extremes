#
# This file includes all functions used for running the ABC algorithm
#
library(SpatialExtremes)
library(parallel)
library(gridExtra)
library(usedist)
library(combinat)
library(interp)
library(easyNCDF)

#' Calculate distance function
#' @description Calculates absolute distance between two vectors
#'
#' @param a Vector one
#' @param b Vector two
#'
#' @return Distance scalar
distance_function <- function(a,b){
  return(sum(abs(a-b)))
}


#' Calculate tripletwise distance
#' @description Calculates tripletwise distance, as specified in the ABC algorithm
#'
#' @param v1 Set of triplets 1
#' @param v2 Set of triplets 2
#'
#' @return Distance scalar
triplet_distance <- function(v1,v2){
  a <- v1
  b <- permn(v2)
  dis <- sapply(b, FUN = distance_function, a)
  return(min(dis))
}


#' Calculate tripletwise extremal coefficient
#'
#' @param idx Index of the triplets of the grid of the process
#' @param data Observations of the max-stable process 
#'
#' @return Scalar: Tripletwise extremal coefficient
triplet_ext_coef <- function(idx, data){
  n <- dim(data)[1]
  x_1 <-data[,idx[1]]
  x_2 <-data[,idx[2]]
  x_3 <-data[,idx[3]]
  max <- apply(cbind(x_1,x_2,x_3), FUN = max, MARGIN = 1)
  return(n/sum(1/max))
}

#' Get data
#' @description Based on the index obtains single sample from (global) test data. If needed the process will be downsampled using
#' bilinear interpolation
#'
#' @param i Index of the sample
#' @param length The length of the grid
#' @param downsample_size The length of the downsampled grid
#' @param type Whether to work with the full or interpolated data
#'
#' @return Sample of max-stable process
get_data <- function(i, length = 30, downsample_size = 5, type = "full"){
  data <- test_data[,i]
  if(type != "full"){
    #If interpolation
    data_transformed <- array(data, dim = c(length, length))
    field_small <- bilinear.grid(x = x, y = x, z = data_transformed, nx = downsample_size, ny = downsample_size)
    data_small <- array(field_small$z, dim = c(1, downsample_size**2))
    return(data_small)
  }else{
    result <- array(data, dim = c(1, length(data)))
    return(result)
  }
}


#' Get clusters
#' @description Calculates the clusters of the tripletwise distances.
#'
#' @param grid The underlying grid of the max-stable process
#' @param n_stations The number of stations/observations in the process
#' @param n_cluster The number of clusters to extract
#' @param method Whether to use all triplets or only a sampled subset
#' @param approx_dim The number of triplets to use in the sampling case
#'
#' @return The index of the triplets, as well as the cluster it belongs to.
get_clusters <- function(grid, n_stations, n_cluster = 100, method = "full", approx_dim = NULL){
  # Calculate distance matrix for grid
  if(method == "full"){
    triplets <- array(data = 0, dim = c(choose(n_stations,3),3))
    #Fill triplets
    index_comb <- combn(n_stations,3)
    for (i in 1:choose(n_stations,3)){
      idx = index_comb[,i]
      triplets[i,1] <- dist((grid[idx[1],]-grid[idx[2],]))
      triplets[i,2] <- dist((grid[idx[1],]-grid[idx[3],]))
      triplets[i,3] <- dist((grid[idx[2],]-grid[idx[3],]))
    }
  }else{
    #Approximate triplets with sampling
    triplets <- array(data = 0, dim = c(approx_dim, 3))
    index_comb <- array(data = 0, dim = c(3,approx_dim))
    for (i in 1:approx_dim){
      idx <- sample(x = n_stations, size = 3, replace = FALSE)
      index_comb[,i] <- idx
      triplets[i,1] <- dist(rbind(grid[idx[1],],grid[idx[2],]))
      triplets[i,2] <- dist(rbind(grid[idx[1],],grid[idx[3],]))
      triplets[i,3] <- dist(rbind(grid[idx[2],],grid[idx[3],]))
    }
  }
  
  #Compute distance matrix
  dist_matrix <- dist_make(triplets, triplet_distance)
  #Clustering
  cluster <- hclust(dist_matrix, method = "ward.D")
  members <- cutree(cluster, k = n_cluster)
  return(list("members" = members, "index" = index_comb))
}


#' Get ABC sample
#' @description Given a pair of parameters, the method simulates max-stable processes and calculates the
#' distance/difference to the observed process.
#'
#' @param params Parameter pair of range and smoothness
#' @param n Number of fields to simulate for comparison
#' @param grid The underlying grid of the max-stable process
#' @param model The max-stable model
#' @param memb A matrix indicating the clustering of the triplets
#' @param index_comb All possible triplets on the grid
#' @param theta The Tripletwise extremal coefficient of the original process
#'
#' @return The distance metric for the specific sample
get_abc_sample <- function(params, n, grid, model, memb, index_comb, theta){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  if (model=="brown"){
    sim_field <- rmaxstab(n = n, coord = grid, cov.mod = model, range = range, smooth = smooth)
  }else{
    sim_field <- rmaxstab(n = n, coord = grid, cov.mod = model, nugget = 0,  range = range, smooth = smooth)
  }
  #Calculate ext coeff
  ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, sim_field)
  #Aggregate per cluster
  theta_est <- aggregate(ext_coeff, by = list(memb), FUN = mean)
  #Calculate distance
  dist <- distance_function(theta,theta_est)
  return(dist)
}


#' Run ABC sampling
#' @description Runs the abc sampling algorithm for a specified number of simulations and outputs 500 particles.
#'
#' @param data The observation of the max-stable process
#' @param grid The underlying grid of the max-stable process
#' @param cluster_res The cluster belonging of each triplet
#' @param model The max-stable model
#' @param n_sim Number of simulations
#' @param n_cores Number of cores to run in parallel
#' @param n_sim_each Number of simulations per abc sample
#'
#' @return 500 estimated parameters with corresponding distance measure
run_abc_sampling <- function(data, grid, cluster_res, model, n_sim,
                             n_cores = 20, n_sim_each = 50){
  memb <- cluster_res$members
  index_comb <- cluster_res$index
  
  #Calculate extremal coefficient
  ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, data)
  #Aggregate per cluster
  theta_true <- aggregate(ext_coeff, by = list(memb), FUN = mean)
  
  #Generate sampling parameters
  smooth <- runif(n = n_sim, min = 0.3, max = 1.8)
  range <- runif(n = n_sim, min = 0.5, max = 5)
  test_params <- cbind(range, smooth)
  
  #Calculate parallel ABC
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('get_abc_sample', 'triplet_ext_coef', 'distance_function'))
  clusterEvalQ(cl, library(SpatialExtremes))
  dist_est <- parApply(cl, test_params, MARGIN = 1, FUN = get_abc_sample,
                       n_sim_each, grid, model, memb, index_comb, theta_true)
  stopCluster(cl)
  result <- cbind(test_params, dist_est)
  
  #Filter results
  # Choose quantile such that 500 particles remain
  q_filter <- 500/n_sim
  q <- quantile(result[,"dist_est"], q_filter)
  result <- result[result[, "dist_est"] <= q,]
  return(result)
}



#' ABC wrapper
#' @description Wrapper function to run the ABC method across a whole test dataset
#'
#' @param index Index of the sample in the dataset
#' @param grid Underlying grid of the max-stable process
#' @param model Max-stable model
#' @param n_sim Number of simulations
#' @param n_cores Number of cores to run in parallel
#' @param n_sim_each Number of simulations per abc sample
#' @param interp Whether to interpolate the data or use the full data
#'
#' @return 500 ABC samples for each observation
abc_wrapper <- function(index, grid, model, n_sim,
                        n_cores = 20, n_sim_each = 50, interp = "full"){
  # Get clusters
  cluster_res <- get_clusters(grid = grid, n_stations = dim(grid)[1], method = "full")
  
  # Measure time
  t1 <- Sys.time()
  data <- get_data(index, interp)
  result <- run_abc_sampling(data = data, grid = grid, cluster_res = cluster_res, model = model,
                             n_sim = n_sim, n_cores = n_cores, n_sim_each = n_sim_each)
  print(paste0("Total time for sample  ",index," : ", Sys.time()-t1))
  return(result)
}