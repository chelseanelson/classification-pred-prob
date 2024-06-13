# Classification Prediction Problem - Round 1 
# Define and fit baseline logistic regression model 

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
load(here("recipes/round_1/baseline_rec.rda"))

# model specifications ----
logistic_model <-
  logistic_reg(mode = "classification") %>% 
  set_engine("glm")

# define workflows ---
logistic_wflow <-
  workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(baseline_rec)

# fit workflows/models
keep_wflow <- control_resamples(save_workflow = TRUE)

fit_logistic_1 <-
  fit_resamples(
    logistic_wflow,
    resamples = airbnb_folds,
    control = keep_wflow
  )

# write out results (fitted/trained workflows) ----
save(fit_logistic_1, file = here("results/round_1/fitted_tuned_models/fit_logistic_1.rda"))
