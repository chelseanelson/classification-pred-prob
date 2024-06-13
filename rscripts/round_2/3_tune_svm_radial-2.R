# Classification Prediction Problem - Round 2 
# Define, fit and tune feature engineered SVM Radial model 
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
load(here("recipes/round_2/fe_rec.rda"))

# model specifications ----
svm_radial_model <-
  svm_rbf(
    mode = "classification",
    cost = tune(),
    rbf_sigma = tune()
  ) %>%
  set_engine("kernlab")

# define workflows ---
svm_radial_wflow <-
  workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(fe_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(svm_radial_model)

# change hyperparameter ranges 
svm_radial_params <- extract_parameter_set_dials(svm_radial_model) %>%
  update(
    cost = cost(range = c(2,40))
  )

rec_params <- extract_parameter_set_dials(fe_rec) %>%
  update(
    num_comp = num_comp(c(10,20))
  )

all_params <- bind_rows(svm_radial_params, rec_params)

# build tuning grid 
svm_radial_grid <- grid_regular(all_params, level = 10)

# tune workflows/models ----
# set seed 
set.seed(1292)
tuned_svm_radial_2 <-
  svm_radial_wflow %>% 
  tune_grid(
    airbnb_folds,
    grid = svm_radial_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_svm_radial_2, file = here("results/round_2/fitted_tuned_models/tuned_svm_radial_2.rda"))