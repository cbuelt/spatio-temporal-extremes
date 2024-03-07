library(SpatialExtremes)
library(lattice)
library(parallel)
library(gridExtra)

#' Get weights for pairwise likelihood
#' @description A weight matrix is constructed (poorly implemented using a for loop) that includes weights for the
#' pairwise likelihood approach
#'
#' @param grid Grid on which the max-stable process is defined
#' @param length Shape of the process
#'
#' @return Weight matrix that can be used for the fitmaxstab function
get_weights <- function(grid, length, cutoff = 5){
  weights <- array(data = NA, dim = choose(length ** 2, 2))
  cnt <- 1
  for (i in 1:(length ** 2 - 1)) {
    for (j in (i + 1):(length ** 2)) {
      if (dist(grid[i, ] - grid[j, ]) <= cutoff) {
        weights[cnt] <- 1
      } else{
        weights[cnt] <- 0
      }
      cnt <- cnt + 1
    }
  }
  return(weights)
}


#' Fit a max-stable process, with given initial parameters
#' @description fits a maxstable process to data with the pairwise likelihood approach given some initial parameters.
#' The process is repeated several times and `n_out` estimations with the highest likelihood are returned.
#'
#' @param initial_parajms Initial parameters
#' @param data Observations from a max-stable process
#' @param grid Underlying grid of the max-stable process
#' @param weights Weights used in fitting the pairwise likelihood
#' @param n_out Number of parameter pairs to ouput
#'
#' @return
run_start_values <- function(initial_params, data, grid, weights, n_out) {
  #Get number of parameters and create lists for results
  n <- dim(initial_params)[1]
  mylist <- vector(mode = "list", length = n)
  names(mylist) <- seq(1, n)
  ll_list <- array(data = 0, dim = n)
  
  #Run optimization
  for (i in 1:n) {
    if (model == "brown") {
      mylist[[i]] <-
        fitmaxstab(
          data = data,
          coord = grid,
          cov.mod = "brown",
          method = "L-BFGS-B",
          start = list("range" = initial_params[i, 1], "smooth" = initial_params[i, 2]),
          weights = weights
        )
    } else{
      mylist[[i]] <-
        fitmaxstab(
          data = data,
          coord = grid,
          cov.mod = model,
          method = "L-BFGS-B",
          start = list(
            "nugget" = 0,
            "range" = initial_params[i, 1],
            "smooth" = initial_params[i, 2]
          ),
          weights = weights
        )
    }
    ll_list[i] <- mylist[[i]]$logLik
  }
  #Get parameters with highest likelihood
  index <-
    sort(ll_list, decreasing = TRUE, index.return = TRUE)$ix[1:n_out]
  
  #Get new starting values
  param_dim <- length(mylist[[i]]$param)
  new_params <- array(data = 0, dim = c(n_out, param_dim))
  for (i in 1:n_out) {
    new_params[i, ] <- mylist[[i]]$fitted.values
  }
  return(new_params)
}


#' Wrapper for running pairwise likelihood in parallel
#' 
#' @description Wrapper for running pairwise likelihood in parallel. Simulations are run on several random initial parameters and the 
#' estimations with the highest likelihood are the output.
#'
#' @param i Index of the current data from the data set
#' @param data The data set with observations of max-stable processes
#' @param grid The underlying grid of the max-stable process
#' @param weights The weight matrix used for the pairwise likelihood
#' @param n_sim The number of simulations
#' @param n_top The number of estimations to keep in the pre-estimation
#'
#' @return
#' @export
#'
#' @examples
apply_mle <-
  function(i,
           data,
           grid,
           weights,
           n_sim = 20,
           n_top = 5) {
    data_subset <- array(rep(data[, i], each = 3), dim = c(3, length ** 2))
    
    # Choose parameter range
    range_seq = c(0.5, 5)
    smooth_seq = c(0.3, 1.8)
    
    # Simulate parameters from uniform distribution
    range <- runif(n_sim, min = range_seq[1], max = range_seq[2])
    smooth <- runif(n_sim, min = smooth_seq[1], max = smooth_seq[2])
    params <- cbind(range, smooth)
    
    #Pre-estimation
    base_params <-
      run_start_values(params, data_subset, grid, weights, n_top)
    #Final estimation
    final_param <-
      run_start_values(base_params, data_subset, grid, weights, 1)
    return(final_param)
  }


