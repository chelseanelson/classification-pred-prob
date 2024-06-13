# Classification Prediction Problem - Round 1 
# Define, fit and tune feature engineered boosted trees model 
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
load(here("recipes/round_1/fe_nonpara_rec.rda"))

# model specifications ----
bt_model <- 
  boost_tree(
    mode = "classification",
    mtry = tune(),
    min_n = 1,
    learn_rate = tune(),
    trees = tune()
  ) %>% 
  set_engine("xgboost")

# define workflows ---
bt_wflow <-
  workflow() %>% 
  add_model(bt_model) %>%
  add_recipe(fe_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(bt_model)

# change hyperparameter ranges 
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(mtry = mtry(c(14,35)),
         learn_rate = learn_rate(range = c(-5, -0.2)),
         trees = trees(range = c(900, 1500))) 

# build tuning grid 
bt_grid <- grid_regular(bt_params, 
                        levels = 8)

# tune workflows/models ----
# set seed 
set.seed(1483)

tuned_bt_2 <-
  bt_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = bt_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_bt_2, file = here("results/round_1/fitted_tuned_models/tuned_bt_2.rda"))
