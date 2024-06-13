# Classification Prediction Problem - Round 2 
# Define, fit and tune baseline SVM Polynomial model 
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
load(here("data/model_data/round_2/airbnb_folds.rda"))

# load pre-procesing/feature engineering recipe
load(here("recipes/round_2/baseline_rec.rda"))

# model specifications ----
svm_poly_model <-
  svm_poly(
    mode = "classification",
    cost = tune(),
    degree = tune(),
    scale_factor = tune()
  ) %>%
  set_engine("kernlab")

# define workflows ---
svm_poly_wflow <-
  workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(baseline_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(svm_poly_model)

# change hyperparameter ranges 
svm_poly_params <- extract_parameter_set_dials(svm_poly_model)

# build tuning grid 
svm_poly_grid <- grid_regular(svm_poly_params, levels = 8)

# tune workflows/models ----
# set seed 
set.seed(5903)
tuned_svm_poly_1 <-
  svm_poly_wflow %>% 
  tune_grid(
    airbnb_folds,
    grid = svm_poly_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_svm_poly_1, file = here("results/round_2/fitted_tuned_models/tuned_svm_poly_1.rda"))