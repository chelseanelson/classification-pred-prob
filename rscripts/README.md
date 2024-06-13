## Classification Prediction Problem - Rscripts

### Documents 

`0_data_cleaning.R` : R script containing the intial data cleaning of the raw dataset

### Folders

`round_1/` : contains the associated R scripts for round 1 model building 

`round_2/` : contains the associated R scripts for round 2 model building 

`round_3/` : contains the associated R scripts for round 3 model building 
 
## Usage

If you want to explore or run the code, I would recommend starting with 
`rscripts/0_data_cleaning.R` to understand the distribution and layout of the 
original data and then running in order after that in order to see the 
development and creation of the different types of models. I would finish off
by looking at `rscripts/4_model_analysis.R` through 
`rscripts/6_assess_final_model.R` to see how each model performs for each of the different rounds, 
looking at which model type has the best roc_auc value. Otherwise, all of the associated model 
data, fitted data, tables of the performance analyses, and the recipes in which
I created can be found in their respective rscripts. 
