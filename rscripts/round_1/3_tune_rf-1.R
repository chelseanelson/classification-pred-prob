# Classification Prediction Problem - Round 1
# Define, fit and tune baseline random forest model 
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
load(here("data/model_data/round_1/airbnb_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/round_1/baseline_nonpara_rec.rda"))

# model specifications ----
rf_model <-
  rand_forest(
    mode = "classification",
    mtry = tune(),
    min_n = tune(),
    trees = 1000
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
    mtry = mtry(range = c(1, 24)),
    min_n = min_n(range = c(1,6))
  )

# build tuning grid 
rf_grid <- grid_regular(rf_params, levels = 6)

# tune workflows/models ----
# set seed 
set.seed(14353)

tuned_rf_1 <-
  rf_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = rf_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_rf_1, file = here("results/round_1/fitted_tuned_models/tuned_rf_1.rda"))
