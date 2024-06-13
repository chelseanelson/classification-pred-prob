# Classification Prediction Problem - Round 3
# Define, fit and tune baseline Boosted Tree model 
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
  add_recipe(baseline_nonpara_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(bt_model)

# change hyperparameter ranges 
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(mtry = mtry(c(20,30)),
         learn_rate = learn_rate(range = c(-5, 0.2)),
         trees = trees(range = c(1000, 1300)))

# build tuning grid 
bt_grid <- grid_regular(bt_params, 
                        levels = 10)

# tune workflows/models ----
# set seed 
set.seed(14718)

tuned_bt_1 <-
  bt_wflow %>%
  tune_grid(
    airbnb_folds,
    grid = bt_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_bt_1, file = here("results/round_3/fitted_tuned_models/tuned_bt_1.rda"))