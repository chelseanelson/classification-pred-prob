# Classification Prediction Problem - Round 1 
# Define and fit naive bayes model 

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(discrim)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data/model_data/round_1/airbnb_folds.rda"))

# load pre-procesing/feature engineering recipe 
load(here("recipes/round_1/baseline_nbayes_rec.rda"))

# model specifications ----
nbayes_model <-
  naive_Bayes(mode = "classification") %>% 
  set_engine("klaR")

# define workflows ---
nbayes_wflow <-
  workflow() %>% 
  add_model(nbayes_model) %>%
  add_recipe(baseline_nbayes_rec)

# fit workflows/models
keep_wflow <- control_resamples(save_workflow = TRUE)

fit_nbayes <-
  fit_resamples(
    nbayes_wflow,
    resamples = airbnb_folds,
    control = keep_wflow
  )

# write out results (fitted/trained workflows) ----
save(fit_nbayes, file = here("results/round_1/fitted_tuned_models/fit_nbayes.rda"))