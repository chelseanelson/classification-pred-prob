# Classification Prediction Problem - Round 3
# Define, fit and tune baseline Random Forest model 2
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data/model_data/round_3/airbnb_folds.rda"))

# load pre-processing/feature engineering recipe
load(here("recipes/round_3/baseline_nonpara_rec.rda"))

# model specifications ----
rf_model <-
  rand_forest(
    mode = "classification",
    mtry = tune(),
    min_n = 1,
    trees = tune()
  ) %>%
  set_engine("ranger")

# define workflows ---
rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(baseline_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(rf_model)

# change hyperparameter ranges 
rf_params <- extract_parameter_set_dials(rf_model) %>%
  update(
    mtry = mtry(range = c(25,40)),
    trees = trees(range = c(1020, 1350))
  )

# build tuning grid 
rf_grid <- grid_regular(rf_params, levels = 10)

# tune workflows/models ----
# set seed 
set.seed(23501)

tuned_rf_2 <-
  rf_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = rf_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_rf_2, file = here("results/round_3/fitted_tuned_models/tuned_rf_2.rda"))
