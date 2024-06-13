# Classification Prediction Problem - Round 2
# Setup pre-processing/recipes

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# realized that threshold for step_other was 0.5, maybe it should be lower like 0.05

# handle common conflicts 
tidymodels_prefer()

## load in training data 
load(here("data/model_data/round_2/airbnb_train.rda"))

# build recipes ----

## recipe 1 (baseline) ----

### variation 1 (parametric)

baseline_rec <- recipe(host_is_superhost ~ ., data = airbnb_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.5) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

### variation 2 (non-parametric)
baseline_nonpara_rec <- 
  recipe(host_is_superhost ~ ., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>% 
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.5) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())

# check recipes 
baseline_rec %>% 
  check_missing(all_predictors()) %>%
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

baseline_nonpara_rec %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

## recipe 2 (feature engineering) ----

### variation 1 (parametric)

fe_rec <- recipe(host_is_superhost ~ ., data = airbnb_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = tune()) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

### variation 2 (non-parametric)

fe_nonpara_rec <- 
  recipe(host_is_superhost ~ ., data = airbnb_train) %>%
  update_role(id, new_role = "id") %>% 
  step_date(all_date_predictors(), keep_original_cols = FALSE) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = tune()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

# check recipes 
fe_rec %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

fe_nonpara_rec %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

# write out recipes
save(baseline_rec, file = here("recipes/round_2/baseline_rec.rda"))
save(baseline_nonpara_rec, file = here("recipes/round_2/baseline_nonpara_rec.rda"))
save(fe_rec, file = here("recipes/round_2/fe_rec.rda"))
save(fe_nonpara_rec, file = here("recipes/round_2/fe_nonpara_rec.rda"))
