# Classification Prediction Problem - Round 2 
# Define, fit and tune feature engineered MARS model 
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
mars_model <-
  mars(mode = "classification",
       num_terms = tune(),
       prod_degree = 6
  ) %>% 
  set_engine("earth")

# define workflows ---
mars_wflow <-
  workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(fe_rec)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(mars_model)

# change hyperparameter ranges 
mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(
    num_terms = num_terms(range = c(25,50))
  )

rec_params <- extract_parameter_set_dials(fe_rec) %>%
  update(
    num_comp = num_comp(c(10,20))
  )

all_params <- bind_rows(mars_params, rec_params)

# build tuning grid 
mars_grid <- grid_regular(all_params, level = 25)

# tune workflows/models ----
# set seed 
set.seed(3432)
tuned_mars_2 <-
  mars_wflow %>% 
  tune_grid(
    airbnb_folds,
    grid = mars_grid,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(tuned_mars_2, file = here("results/round_2/fitted_tuned_models/tuned_mars_2.rda"))