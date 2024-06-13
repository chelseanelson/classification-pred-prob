# Classification Prediction Problem - Round 3
# Train final model
# Best Model: Baseline Random Forest 2
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

# best model: 

# load tuned and training data 
load(here("results/round_3/fitted_tuned_models/tuned_rf_2.rda"))
load(here("data/model_data/round_3/airbnb_train.rda"))

# finalize workflow ----
final_wflow <-
  tuned_rf_2 %>%
  extract_workflow(tuned_rf_2) %>%
  finalize_workflow(select_best(tuned_rf_2, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(0902)
final_fit <- fit(final_wflow, airbnb_train)

# write out fitted data 
save(final_fit, file = here("results/round_3/final_fit.rda"))