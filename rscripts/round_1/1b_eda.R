# Classification Prediction Problem - Round 1 
# Exploratory Data Analysis on subsection of training data 
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data 
load(here("data/model_data/round_1/airbnb_train.rda"))

# skim training data
airbnb_train %>% skimr::skim_without_charts()

# host_location, host_response_time, host_response_rate, host_acceptance_rate, host_neighbourhood,
# bathrooms_text, beds, reviews_scores_rating, review_scores_accuracy, review_scores_cleanliness,
# review_scores_checkin, review_scores_communication, review_scores_location, 
# review_scores_value, reviews_per_month

# random sample of training data 
# set seed 
set.seed(348902)
airbnb_eda <- airbnb_train %>% slice_sample(prop = .8)
airbnb_eda %>% skimr::skim_without_charts()

# univariate analysis ----

## continuous variables ----

numeric_columns <- airbnb_eda %>% select_if(is.numeric)

# Map function to create histograms for each numeric variable
histograms <- map(names(numeric_columns), ~ ggplot(data = airbnb_eda, mapping = aes(x = !!sym(.))) +
                    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
                    labs(title = paste("Histogram of", tools::toTitleCase(gsub("_", " ", .))), x = tools::toTitleCase(gsub("_", " ", .)), y = "Count") +
                    scale_x_continuous(labels = scales::number_format()) + theme_minimal())


## discrete variables ----

factor_and_logical_columns <- airbnb_eda %>% select_if(is.factor) %>% 
  select(-c(host_is_superhost, neighbourhood_cleansed, host_location, host_neighbourhood, host_acceptance_rate, host_response_rate, host_location, property_type, bathrooms_text, id))

barplots <- map(names(factor_and_logical_columns), ~ ggplot(data = airbnb_eda, mapping = aes(x = !!sym(.))) + geom_bar(fill = "skyblue", color = "black") + labs(title = paste("Distribution of", tools::toTitleCase(gsub("_", " ", .))), x = tools::toTitleCase(gsub("_", " ", .)), y = "Count") + theme_minimal() + 
                  scale_x_discrete(labels = function(x) str_to_title(x)) + 
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5))
  

# write out univariate plots
for (i in seq_along(histograms)) {
  filename <- paste0("figures/univariate/figure-", i, "_univariate_continuous.png")
  ggsave(filename = filename, plot = histograms[[i]])
}

for (i in seq_along(barplots)) {
  filename <- paste0("figures/univariate/figure-", i, "_univariate_discrete.png")
  ggsave(filename = filename, plot = barplots[[i]])
}

# bivariate analysis ----
bivariate_response_continuous <- map(names(numeric_columns), ~ ggplot(data = airbnb_eda, mapping = aes(x = host_is_superhost, y = !!sym(.))) + geom_boxplot(fill = "skyblue", color = "black") + labs(title = paste(tools::toTitleCase(gsub("_", " ", .)), "by SuperHost Status"), x = "SuperHost Status", y = tools::toTitleCase(gsub("_", " ", .))) + theme_minimal() + scale_x_discrete(labels = function(x) str_to_title(x)))

bivariate_response_discrete <- map(names(factor_and_logical_columns), ~ ggplot(data = airbnb_eda, mapping = aes(fill = host_is_superhost, x = !!sym(.))) + geom_bar(color = "black", position = "dodge") + labs(title = paste(tools::toTitleCase(gsub("_", " ", .)), "by SuperHost Status"), fill = "SuperHost Status", x = tools::toTitleCase(gsub("_", " ", .)), y = "Count") + theme_minimal() + scale_x_discrete(labels = function(x) str_to_title(x)) + 
  scale_fill_manual(values = c("pink", "skyblue")) + 
    geom_text(stat = "count", aes(label = ..count..),position = position_dodge(width = 0.9), vjust = -0.5))

# write out bivariate plots 

for (i in seq_along(bivariate_response_discrete)) {
  filename <- paste0("figures/bivariate/figure-", i, "_bivariate_response_discrete.png")
  ggsave(filename = filename, plot = bivariate_response_discrete[[i]])
}

for (i in seq_along(bivariate_response_continuous)) {
  filename <- paste0("figures/bivariate/figure-", i, "_bivariate_response_continuous.png")
  ggsave(filename = filename, plot = bivariate_response_continuous[[i]])
}
