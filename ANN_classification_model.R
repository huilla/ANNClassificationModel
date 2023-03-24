###############################################################
# Data Science - Assignment 2 (Neural Networks)
# Autumn semester 2022
###############################################################

## Task description

# Provide a classification model for credit card customers using neuronal networks
# (the dataset was provided by the professor)

##### The best model using 10-fold validation approach #######

## libraries
library(tidyverse) # handy functions such as select and mutate
library(fastDummies) # dummy variables
library(keras) # ANN
library(tensorflow) # ANN
library(tidymodels) # training/test data split

## read data
NNraw <-  read.csv("Dataset-part-2.csv")

## explore data
str(NNraw) # 67614 obs. of  19 variables, class int/num = 10 / chr = 9 variables
summary(NNraw)

###### PREPARE DATA ######

# we need numeric/integer input variables with similar scale

NNdata <- NNraw

## remove ID and FLAG_MOBIL
NNdata$ID <- NULL # includes no useful information
unique(NNdata$FLAG_MOBIL) # each row has the same value (1), no NAs -> everyone has a mobile phone
NNdata$FLAG_MOBIL <- NULL # we assume that the data is representative = the value for this column is 1 also in the new data -> safe to remove

## deal with missing values
# OCCUPATION_TYPE is the only column that has NA values
unique(NNdata$OCCUPATION_TYPE) # 18 categories plus NA values
mean(is.na(NNdata$OCCUPATION_TYPE)) # 31 % of data is missing -> Neural networks cannot handle any NA values
# View(NNdata[is.na(NNdata$OCCUPATION_TYPE), ]) # view NAs
occupation_NA <- NNdata %>% filter(is.na(OCCUPATION_TYPE)) # save as an object to explore rows with NAs
occupation <- NNdata %>% filter(!is.na(OCCUPATION_TYPE)) # save as an object to explore rows without NAs
unique(occupation_NA$NAME_INCOME_TYPE) # "Pensioner", "State servant", "Working", "Commercial associate", "Student" 
unique(occupation$NAME_INCOME_TYPE) # "Pensioner", "State servant", "Working", "Commercial associate", "Student"
summary(occupation_NA$DAYS_EMPLOYED) # min -16365, median 365243, max 365243 -> large values (365243/365 = over 1000 years)
summary(occupation$DAYS_EMPLOYED) # min -17531, median -1733, max -12
sum(NNdata$DAYS_EMPLOYED > 0) # 11855 observations
sum(NNdata$DAYS_EMPLOYED > 0 & NNdata$NAME_INCOME_TYPE == "Pensioner") # 11855 observations
sum(NNdata$NAME_INCOME_TYPE == "Pensioner") # 11982
sum(NNdata$NAME_INCOME_TYPE == "Pensioner" & is.na(NNdata$OCCUPATION_TYPE)) # 11877
sum(NNdata$NAME_INCOME_TYPE == "Pensioner" & !is.na(NNdata$OCCUPATION_TYPE)) # 105
# View(NNdata[NNdata$DAYS_EMPLOYED > 0, ]) # view positive values --> 365243 means unemployed, there are no other positive values
sum(NNdata$DAYS_EMPLOYED == 0) # 0 so no zeros

# interpretation:
# if DAYS_EMPLOYED is 365243, the person is in pension and OCCUPATION_TYPE is NA (11855 observations)
# in total there are 20699 observations where occupation type is NA, out of those 11877 belong to people who are in pension
# There are 105 cases where someone is in pension but occupation type is not NA
# if a person is not working DAYS_EMPLOYED is set to 365243. This would be over 1000 years.
# The value is very high compared to other values and might make our model think these values are better.
# As a solution we will create a new variables "employed" and "unemployed" and after that set 365243 to 0 in DAYS_EMPLOYED.

## Recode missing values in OCCUPATION_TYPE
NNdata$OCCUPATION_TYPE[is.na(NNdata$OCCUPATION_TYPE) & NNdata$NAME_INCOME_TYPE == "Pensioner"] <- "Pension" # person is in pension
NNdata$OCCUPATION_TYPE[is.na(NNdata$OCCUPATION_TYPE)] <- "Missing" # means person is working but occupation type is not available
sum(is.na(NNdata$OCCUPATION_TYPE)) # no missing values
unique(NNdata$OCCUPATION_TYPE) # 20 categories

## Create a new variables "employed" and "unemployed"
NNdata$employed <- NNdata$DAYS_EMPLOYED
NNdata$employed <- ifelse(NNdata$DAYS_EMPLOYED != 365243, 1, NNdata$employed) # if value is not 365243, the person is employed (1)
NNdata$employed <- ifelse(NNdata$DAYS_EMPLOYED == 365243, 0, NNdata$employed) # if value is 365243, the person is not employed (0)

NNdata$unemployed <- NNdata$DAYS_EMPLOYED
NNdata$unemployed <- ifelse(NNdata$DAYS_EMPLOYED != 365243, 0, NNdata$unemployed) # if value is not 365243, the person is not unemployed (0)
NNdata$unemployed <- ifelse(NNdata$DAYS_EMPLOYED == 365243, 1, NNdata$unemployed) # if values is 365243, the person is unemployed (1)

## Deal with high values in DAYS_EMPLOYED
NNdata$DAYS_EMPLOYED[NNdata$DAYS_EMPLOYED == 365243] <- 0

## create dummy variables
NNdata <- dummy_cols(NNdata,
                     select_columns = c("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                                        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"),
                     remove_selected_columns = TRUE)

## DAYS_BIRTH and DAYS_EMPLOYED
NNdata$DAYS_BIRTH <- abs(NNdata$DAYS_BIRTH) # from negative to positive
NNdata$DAYS_EMPLOYED <- abs(NNdata$DAYS_EMPLOYED) # from negative to positive

## create a function that converts values to a range between 0 and 1
range_function <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

## scale to 0/1
NNdata$CNT_CHILDREN <- range_function(x = NNdata$CNT_CHILDREN)
NNdata$AMT_INCOME_TOTAL <- range_function(x = NNdata$AMT_INCOME_TOTAL)
NNdata$DAYS_BIRTH <- range_function(x = NNdata$DAYS_BIRTH)
NNdata$DAYS_EMPLOYED <- range_function(x = NNdata$DAYS_EMPLOYED)
NNdata$CNT_FAM_MEMBERS <- range_function(x = NNdata$CNT_FAM_MEMBERS)

dim(NNdata) # 67614 observation, 58 variables

## status (output variable)
unique(NNdata$status) # categories: "0" "2" "X" "C" "1" "4" "5" "3"
train_labels <- select(NNdata, c(status)) # new data frame with only status variable
train_labels <- dummy_cols(train_labels, select_columns = c("status"), remove_selected_columns = TRUE) # make 8 new dummy columns

## Prepare data
train_labels <- data.matrix(train_labels) # matrix to feed to the neural network
train_data <- select(NNdata, -c(status)) # remove status variable
train_data <- data.matrix(train_data) # matrix to feed to the neural network

## check variables
summary(train_data) # all input variable values are between 0 and 1, some floating numbers, some Booleans
str(train_data) # num [1:67614, 1:57] 0.1053 0 0 0.0526 0 ...
summary(train_labels)
str(train_labels) # int [1:67614, 1:8] 1 1 1 0 1 1 1 1 1 1 ...

###### BUILD NETWORK ######

# Because we will need to instantiate the same model multiple times,
# we use a function to construct it.
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 300, regularizer_l1(l = 0.01), activation = "relu", input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 200, regularizer_l1(l = 0.01), activation = "relu") %>%
    layer_dense(units = 8, activation = "softmax") 
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate=0.001), 
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
}

## K-fold validation

k <- 10
set.seed(1)
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 239
batch_size <- 512
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  cat("    prepare data for fold #", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- train_data[val_indices,]
  val_labels <- train_labels[val_indices,]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_labels <- train_labels[-val_indices,]
  
  # Build the Keras model (already compiled)
  cat("    build model for fold #", i, "\n")
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  cat("    train model for fold #", i, "\n")
  model %>% fit(partial_train_data, partial_train_labels,
                epochs = num_epochs, batch_size = batch_size, verbose = 0)
  
  # Evaluate the model on the validation data
  cat("    evaluate model for fold #", i, "\n")
  results <- model %>% evaluate(val_data, val_labels, verbose = 0)
  #all_scores <- c(all_scores, results$accuracy)
  all_scores <- c(all_scores, results["accuracy"])
} 

## Evaluate
all_scores
mean(all_scores) # 0.8642146 (average accuracy, this is the accuracy we reported)


##### LEARN AND SAVE THE BEST MODEL #######

## To be sure the whole code is executed again (incl. preprocessing) because we need to do the split the data to training and testing sets.

## libraries
library(tidyverse) # handy functions such as select and mutate
library(fastDummies) # dummy variables
library(keras) # ANN
library(tensorflow) # ANN
library(tidymodels) # training/test data split

## read data
NNraw <-  read.csv("Dataset-part-2.csv")

## explore data
str(NNraw) # 67614 obs. of  19 variables, class int/num = 10 / chr = 9 variables
summary(NNraw)

###### PREPARE DATA ######

# we need numeric/integer input variables with similar scale

NNdata <- NNraw

## remove ID and FLAG_MOBIL
NNdata$ID <- NULL # includes no useful information
unique(NNdata$FLAG_MOBIL) # each row has the same value (1), no NAs -> everyone has a mobile phone
NNdata$FLAG_MOBIL <- NULL # we assume that the data is representative = the value for this column is 1 also in the new data -> safe to remove

## deal with missing values
# OCCUPATION_TYPE is the only column that has NA values
unique(NNdata$OCCUPATION_TYPE) # 18 categories plus NA values
mean(is.na(NNdata$OCCUPATION_TYPE)) # 31 % of data is missing -> Neural networks cannot handle any NA values
# View(NNdata[is.na(NNdata$OCCUPATION_TYPE), ]) # view NAs
occupation_NA <- NNdata %>% filter(is.na(OCCUPATION_TYPE)) # save as an object to explore rows with NAs
occupation <- NNdata %>% filter(!is.na(OCCUPATION_TYPE)) # save as an object to explore rows without NAs
unique(occupation_NA$NAME_INCOME_TYPE) # "Pensioner", "State servant", "Working", "Commercial associate", "Student" 
unique(occupation$NAME_INCOME_TYPE) # "Pensioner", "State servant", "Working", "Commercial associate", "Student"
summary(occupation_NA$DAYS_EMPLOYED) # min -16365, median 365243, max 365243 -> large values (365243/365 = over 1000 years)
summary(occupation$DAYS_EMPLOYED) # min -17531, median -1733, max -12
sum(NNdata$DAYS_EMPLOYED > 0) # 11855 observations
sum(NNdata$DAYS_EMPLOYED > 0 & NNdata$NAME_INCOME_TYPE == "Pensioner") # 11855 observations
sum(NNdata$NAME_INCOME_TYPE == "Pensioner") # 11982
sum(NNdata$NAME_INCOME_TYPE == "Pensioner" & is.na(NNdata$OCCUPATION_TYPE)) # 11877
sum(NNdata$NAME_INCOME_TYPE == "Pensioner" & !is.na(NNdata$OCCUPATION_TYPE)) # 105
# View(NNdata[NNdata$DAYS_EMPLOYED > 0, ]) # view positive values --> 365243 means unemployed, there are no other positive values
sum(NNdata$DAYS_EMPLOYED == 0) # 0 so no zeros

# interpretation:
# if DAYS_EMPLOYED is 365243, the person is in pension and OCCUPATION_TYPE is NA (11855 observations)
# in total there are 20699 observations where occupation type is NA, out of those 11877 belong to people who are in pension
# There are 105 cases where someone is in pension but occupation type is not NA
# if a person is not working DAYS_EMPLOYED is set to 365243. This would be over 1000 years.
# The value is very high compared to other values and might make our model think these values are better.
# As a solution we will create a new variables "employed" and "unemployed" and after that set 365243 to 0 in DAYS_EMPLOYED.

## Recode missing values in OCCUPATION_TYPE
NNdata$OCCUPATION_TYPE[is.na(NNdata$OCCUPATION_TYPE) & NNdata$NAME_INCOME_TYPE == "Pensioner"] <- "Pension" # person is in pension
NNdata$OCCUPATION_TYPE[is.na(NNdata$OCCUPATION_TYPE)] <- "Missing" # means person is working but occupation type is not available
sum(is.na(NNdata$OCCUPATION_TYPE)) # no missing values
unique(NNdata$OCCUPATION_TYPE) # 20 categories

## Create a new variables "employed" and "unemployed"
NNdata$employed <- NNdata$DAYS_EMPLOYED
NNdata$employed <- ifelse(NNdata$DAYS_EMPLOYED != 365243, 1, NNdata$employed) # if value is not 365243, the person is employed (1)
NNdata$employed <- ifelse(NNdata$DAYS_EMPLOYED == 365243, 0, NNdata$employed) # if value is 365243, the person is not employed (0)

NNdata$unemployed <- NNdata$DAYS_EMPLOYED
NNdata$unemployed <- ifelse(NNdata$DAYS_EMPLOYED != 365243, 0, NNdata$unemployed) # if value is not 365243, the person is not unemployed (0)
NNdata$unemployed <- ifelse(NNdata$DAYS_EMPLOYED == 365243, 1, NNdata$unemployed) # if values is 365243, the person is unemployed (1)

## Deal with high values in DAYS_EMPLOYED
NNdata$DAYS_EMPLOYED[NNdata$DAYS_EMPLOYED == 365243] <- 0

## create dummy variables
NNdata <- dummy_cols(NNdata,
                     select_columns = c("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                                        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"),
                     remove_selected_columns = TRUE)

## DAYS_BIRTH and DAYS_EMPLOYED
NNdata$DAYS_BIRTH <- abs(NNdata$DAYS_BIRTH) # from negative to positive
NNdata$DAYS_EMPLOYED <- abs(NNdata$DAYS_EMPLOYED) # from negative to positive

## create a function that converts values to a range between 0 and 1
range_function <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

## scale to 0/1
NNdata$CNT_CHILDREN <- range_function(x = NNdata$CNT_CHILDREN)
NNdata$AMT_INCOME_TOTAL <- range_function(x = NNdata$AMT_INCOME_TOTAL)
NNdata$DAYS_BIRTH <- range_function(x = NNdata$DAYS_BIRTH)
NNdata$DAYS_EMPLOYED <- range_function(x = NNdata$DAYS_EMPLOYED)
NNdata$CNT_FAM_MEMBERS <- range_function(x = NNdata$CNT_FAM_MEMBERS)

dim(NNdata) # 67614 observation, 58 variables

## status (output variable)
unique(NNdata$status) # categories: "0" "2" "X" "C" "1" "4" "5" "3"
NNdata <- dummy_cols(NNdata, select_columns = c("status"), remove_selected_columns = TRUE) # make 8 new dummy columns

## Partition the dataset into training and test set
set.seed(1)
split <- initial_split(NNdata, 0.95) # we do a 95/5 split to have a lot of data for training
train_set <- training(split) # 95% training data
test_set <- testing(split) # 5% test data

train_labels <- select(train_set, c(status_0, status_1, status_2, status_3, status_4, status_5, status_C, status_X)) # only status variables
train_labels <- data.matrix(train_labels) # matrix to feed to the neural network
train_data <- select(train_set, -c(status_0, status_1, status_2, status_3, status_4, status_5, status_C, status_X)) # without status variables
train_data <- data.matrix(train_data) # matrix to feed to the neural network

test_labels <- select(test_set, c(status_0, status_1, status_2, status_3, status_4, status_5, status_C, status_X)) # only status variables
test_labels <- data.matrix(test_labels) # matrix to feed to the neural network
test_data <- select(test_set, -c(status_0, status_1, status_2, status_3, status_4, status_5, status_C, status_X)) # without status variables
test_data <- data.matrix(test_data) # matrix to feed to the neural network

## check variables
summary(train_data) # all input variable values are between 0 and 1, some floating numbers, some Booleans
str(train_data) # num [1:64233, 1:57] 0 0.105 0 0.105 0 ...
str(train_labels) # int [1:64233, 1:8] 1 1 1 0 1 1 1 1 0 1 ...

summary(test_data) # all input variable values are between 0 and 1, some floating numbers, some Booleans
str(test_data) # num [1:3381, 1:57] 0 0 0.105 0 0 ...
str(test_labels) # int [1:3381, 1:8] 1 1 1 1 1 1 1 1 1 1 ...

## save the datasets that were used for the experiments
write.csv(train_set, "2b_train_set.csv")
write.csv(test_set, "2b_test_set.csv")

## Build network

model <- keras_model_sequential() %>% 
  layer_dense(units = 300, regularizer_l1(l = 0.01), activation = "relu", input_shape = c(57)) %>%
  layer_dense(units = 200, regularizer_l1(l = 0.01), activation = "relu") %>%
  layer_dense(units = 8, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(),
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy")
)

history <- model %>% fit(
  train_data,
  train_labels,
  epochs = 239,
  batch_size = 512,
  validation_data = list(test_data, test_labels)
)

# loss: 0.0908 - accuracy: 0.9717 - val_loss: 0.8944 - val_accuracy: 0.8684

## Save the model
save_model_hdf5(model, "2a_final_model.h5")
