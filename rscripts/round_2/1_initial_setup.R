# Classification Prediction Problem - Round 2 
# Initial data checks, data splitting, & data folding 
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load in dataset 
airbnb_data <- read_rds(here("data/train_classification.rds"))

# skim the data
airbnb_data %>% skimr::skim_without_charts()
# there is no na data within our response variable, however a lot of the 
# other variables have na data thus I will have to take care of this within
# the recipe through imputation 

# looking at `host_is_superhost` variable 
airbnb_univariate <- airbnb_data %>% ggplot(aes(host_is_superhost)) + geom_bar(stat = "count", fill = "skyblue", color = "black") + labs(
  title = "Distribution of Superhost Hosts", x = "Host is Superhost", y = "Count") + 
  geom_text(stat = "count", aes(label = after_stat(count)),
            vjust = -0.5) + 
  scale_x_discrete(labels = c("True", "False")) + theme_minimal()

# there is a 1.27:1 ratio thus it is pretty balanced, therefore no upsampling
# or downsampling will have to take place 

# saving figures
#ggsave(here("figures/figure-1.png"), airbnb_univariate)

## set seed for random split 
set.seed(694)

# initial split of the data ----
airbnb_splits <- airbnb_data %>% 
  initial_split(prop = 0.80, strata = host_is_superhost)

airbnb_train <- airbnb_splits %>% training()
airbnb_test <- airbnb_splits %>% testing()

dim(airbnb_train)
dim(airbnb_test)

# folding data (resamples) ----
# set seed 
set.seed(677)
airbnb_folds <- vfold_cv(airbnb_train, v = 5, repeats = 3, strata = host_is_superhost)

# write out split, train, test and folds 
save(airbnb_splits, file = here("data/model_data/round_2/airbnb_splits.rda"))
save(airbnb_train, file = here("data/model_data/round_2/airbnb_train.rda"))
save(airbnb_test, file = here("data/model_data/round_2/airbnb_test.rda"))
save(airbnb_folds, file = here("data/model_data/round_2/airbnb_folds.rda"))