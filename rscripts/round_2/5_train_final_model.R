# Classification Prediction Problem - Round 2
# Train final model
# Best Model: Baseline Boosted Tree Model
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

# best model: Baseline Boosted Tree Model

# load tuned and training data 
load(here("results/round_2/fitted_tuned_models/tuned_bt_1.rda"))
load(here("data/model_data/round_2/airbnb_train.rda"))

# finalize workflow ----
final_wflow <-
  tuned_bt_1 %>%
  extract_workflow(tuned_bt_1) %>%
  finalize_workflow(select_best(tuned_bt_1, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(322)
final_fit <- fit(final_wflow, airbnb_train)

# write out fitted data 
save(final_fit, file = here("results/round_2/final_fit.rda"))