# Classification Prediction Problem - Round 1
# Define, fit and tune baseline neural network model 
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

# load pre-procesing/feature engineering recipe 
load(here("recipes/round_1/baseline_nonpara_rec.rda"))

# model specifications ----
nnet_model <-
  mlp(
    mode = "classification",
    hidden_units = tune(),
    penalty = tune()
  ) %>% 
  set_engine("nnet")

# define workflows ---
nnet_wflow <-
  workflow() %>%
  add_model(nnet_model) %>% 
  add_recipe(baseline_nonpara_rec)

# hyperparameter tuning values ----
nnet_params <- extract_parameter_set_dials(nnet_model)

# build tuning grid 
nnet_grid <- grid_regular(nnet_params, levels = 7)

# tune workflows/models ----
# set seed 
set.seed(110353)
tuned_nnet_1 <- nnet_wflow %>% 
  tune_grid(
    resamples = airbnb_folds,
    grid = nnet_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_nnet_1, file = here("results/round_1/fitted_tuned_models/tuned_nnet_1.rda"))