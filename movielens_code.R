# Outline -----------------------------------------------------------------
# Subject: Recommendation System for PH125.9X Capstone course
# Author: Samuel Bates
#
# This file contains the code used to achieve the results described in
# movielens_report.pdf. It has the following sections:
#   Logging functions:
#     Functions for writing results with timestamps to the console and
#     a log file. Also used to gauge how long various operations take.
#                     
#  Utility functions:
#    Functions to split a dataset into training and test sets, and
#    to calculate the RMSE of a vector of predictions.
#
#  Models using non-genre variables:
#    Includes a description of the model structure and several models.
#
#  Main Program:
#    1. Loads the data.
#    2. Creates the training and validation sets.
#    3. Compares the non-genre models using cross validation and
#         logs the results.
#    4. Creates a genre-based model.
#    5. Checks the combined best non-genre model & genre model on the
#         training set and logs the results.
#    6. Checks the combined best non-genre model & genre model on the
#         validation set and logs the results.
#
#   A few additional notes about the main program:
#   - It periodically saves the environment so that portions can be
#     re-run without starting over from scratch.
#   - It periodically removes large objects so that the analysis does
#     not fail due to running out of memory.
#   - The code was executed on a Windows 7 PC with 6 GB of RAM.
#     Memory issues were frequent, since the training set required
#     1.6 GB of RAM.
#
require(tidyverse)
require(caret)
require(data.table)
require(lubridate)
require(log4r)

options(digits = 5)

## Logging functions ------------------------------------------------------
z__logger <- NULL

#-------------------------------------------------------------
# Set up a simple log with time stamps. The log will write to
# both the console and a log file.
#
log_start <- function() {
  filename <- format(Sys.time(), "logs/%Y%m%d_%H%M%S.txt")
  console_appender <- console_appender(layout = default_log_layout())
  file_appender <- file_appender(filename, append = TRUE,
                                 layout = default_log_layout())
  z__logger <<- log4r::logger(threshold = "INFO",
                              appenders = list(console_appender,
                                               file_appender))
}

#-------------------------------------------------------------
# Write a message to the console and a log file.
log_info <- function(msg) {
  if (is.null(z__logger))
    log_start()
  log4r::info(z__logger, msg)
}

## Utility functions-------------------------------------------------------

#-------------------------------------------------------------
# Partitions a dataset into a test set and training set.
# Parameters:
#   dataset    - A set of movie ratings.
#   test_index - An index of rows for the test set.
# Returns:
#   A list of two sets: train and test. Every rating in
#     dataset is in only one set.
#
get_training_and_test <- function(dataset, test_index) {
  train <- dataset[-test_index, ]
  temp <- dataset[test_index, ]

  # Make sure userId and movieId in test set are also in training set
  test <- temp %>%
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")

  # Add rows removed from test set back into training set
  removed <- anti_join(temp, test, by = colnames(test))
  train <- rbind(train, removed)
  return(list(train = train, test = test))
}

#-------------------------------------------------------------
# Calculates the root mean squared error.
# Parameters:
#   predicted - a vector of predicted values
#   actual    - a vector of values to check against
# Returns:
#   The error.
#
rmse <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}

#-------------------------------------------------------------
# Returns the parameter if it is not NA and 0 otherwise.
# Needed in the models for variables that may not be present.
#
val_or_0 <- function(x)
  ifelse(!is.na(x), x, 0)


## Model generators -------------------------------------------------------
# The following two functions generate models that are tested by cross
# validation. The model structure is described following the generator
# functions.

# Generates a model that is a linear combination of two variables.
# Parameters:
#   model_name - the name of the model
#   var1       - the name of the first variable
#   reg1       - TRUE if the first variable should be regularized,
#                FALSE otherwise
#   var2       - the name of the second variable
#   reg2       - TRUE if the second variable should be regularized,
#                FALSE otherwise
#
model_2var_gen <- function(model_name, 
                           var1, reg1, var2, reg2) {
  list(
    name = model_name,
    train = function(dataset, lambda) {
      mu <- mean(dataset$rating)
      
      e1 <- dataset %>%
        group_by(across(all_of(var1))) %>%
        summarize(e1 = ifelse(reg1,
                              sum(rating - mu) / (n() + lambda),
                              mean(rating - mu)))
      
      e2 <- dataset %>%
        left_join(e1, by = all_of(var1)) %>%
        group_by(across(all_of(var2))) %>%
        summarize(e2 = ifelse(reg2,
                              sum(rating - mu - e1) / (n() + lambda),
                              mean(rating - mu - e1)))
      
      predictfn <- function(dataset) {
        dataset %>%
          left_join(e1, by = all_of(var1)) %>%
          left_join(e2, by = all_of(var2)) %>%
          mutate(pred = mu + val_or_0(e1) + val_or_0(e2)) %>%
          pull(pred)
      }
      return(list(predict = predictfn))
    }
  )
}

# Generates a model that is a linear combination of three variables.
# Parameters:
#   model_name - the name of the model
#   var1       - the name of the first variable
#   reg1       - TRUE if the first variable should be regularized,
#                FALSE otherwise
#   var2       - the name of the second variable
#   reg2       - TRUE if the second variable should be regularized,
#                FALSE otherwise
#   var3       - the name of the third variable
#   reg3       - TRUE if the third variable should be regularized,
#                FALSE otherwise
#
model_3var_gen <- function(model_name, 
                           var1, reg1, var2, reg2, var3, reg3) {
  list(
    name = model_name,
    train = function(dataset, lambda) {
      mu <- mean(dataset$rating)
      
      e1 <- dataset %>%
        group_by(across(all_of(var1))) %>%
        summarize(e1 = ifelse(reg1,
                              sum(rating - mu) / (n() + lambda),
                              mean(rating - mu)))
      
      e2 <- dataset %>%
        left_join(e1, by = all_of(var1)) %>%
        group_by(across(all_of(var2))) %>%
        summarize(e2 = ifelse(reg2,
                              sum(rating - mu - e1) / (n() + lambda),
                              mean(rating - mu - e1)))
      
      e3 <- dataset %>%
        left_join(e1, by = all_of(var1)) %>%
        left_join(e2, by = all_of(var2)) %>%
        group_by(across(all_of(var3))) %>%
        summarize(e3 = ifelse(reg3,
                              sum(rating - mu - e1 - e2) / (n() + lambda),
                              mean(rating - mu - e1 - e2)))
      
      predictfn <- function(dataset) {
        dataset %>%
          left_join(e1, by = all_of(var1)) %>%
          left_join(e2, by = all_of(var2)) %>%
          left_join(e3, by = all_of(var3)) %>%
          mutate(pred = mu + val_or_0(e1) + val_or_0(e2) + val_or_0(e3)) %>%
          pull(pred)
      }
      return(list(predict = predictfn))
    }
  )
}

## Model structure --------------------------------------------------------
# Each of the following objects contains a model used to predict
# movie ratings. Each object is a list containing two elements:
# - a name indicating the model used; and
# - a train() function used to train the model.
# The train() function takes a dataset and a tuning parameter
# and returns a list containing a predict() function to run
# on a dataset.
#

### Regularized(Movie + User) Effects -------------------------------------
model_reg_movie_user <- model_2var_gen(
  "Regularized(Movie + User) Effects",
  "movieId", TRUE, "userId", TRUE)

### Regularized(Movie) + User Effects -------------------------------------
model_user_reg_movie <- model_2var_gen(
  "Regularized(Movie) + User Effects",
  "movieId", TRUE, "userId", FALSE)

### Regularized(Movie + User + Year) Effects ------------------------------
model_reg_movie_user_year <- model_3var_gen(
  "Regularized(Movie + User + Year) Effects",
  "movieId", TRUE, "userId", TRUE, "year", TRUE)

### Regularized(Movie + Year) + User Effects ------------------------------
model_user_reg_movie_year <- model_3var_gen(
  "Regularized(Movie + Year) + User Effects",
  "movieId", TRUE, "userId", FALSE, "year", TRUE)

### Regularized(Movie + User) + Year Effects ------------------------------
model_year_reg_movie_user <- model_3var_gen(
  "Regularized(Movie + User) + Year Effects",
  "movieId", TRUE, "userId", TRUE, "year", FALSE)

### Regularized(Movie) + User + Year Effects ------------------------------
model_user_year_reg_movie <- model_3var_gen(
  "Regularized(Movie) + User + Year Effects",
  "movieId", TRUE, "userId", FALSE, "year", FALSE)

### Regularized(Movie + User) + Age Effects -------------------------------
model_age_reg_movie_user <- model_3var_gen(
  "Regularized(Movie + User) + Age Effects",
  "movieId", TRUE, "userId", TRUE, "age", FALSE)

### Regularized(Movie) + User + Age Effects -------------------------------
model_age_user_reg_movie <- model_3var_gen(
  "Regularized(Movie) + User + Age Effects",
  "movieId", TRUE, "userId", FALSE, "age", FALSE)

### Regularized(Movie + User + Age) Effects -------------------------------
model_reg_movie_user_age <- model_3var_gen(
  "Regularized(Movie + User + Age) Effects",
  "movieId", TRUE, "userId", TRUE, "age", TRUE)

### Regularized(Movie + Age) + User Effects -------------------------------
model_user_reg_movie_age <- model_3var_gen(
  "Regularized(Movie + Age) + User Effects",
  "movieId", TRUE, "userId", FALSE, "age", TRUE)

# Main Program ------------------------------------------------------------


## 1. Load data --------------------------------------------------------------
log_start()

log_info("Loading data...")

records <- readLines("ml-10M100K/ratings.dat")
ratings <- str_split_fixed(records, "\\::", 4)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- as.data.table(ratings) %>%
  mutate(userId = as.numeric(userId),
         movieId = as.numeric(movieId),
         rating = as.numeric(rating),
         timestamp = as.POSIXct(as.numeric(timestamp),
                                tz = "UTC",
                                origin = "1970-01-01 00:00.00 UTC"))
remove(records)

log_info("Ratings data loaded.")

## 2. Create training and test sets ------------------------------------------
log_info("Creating training data...")

# Validation set will be 10% of MovieLens data
suppressWarnings(set.seed(1, sample.kind = "Rounding"))
test_index <- createDataPartition(y = ratings$rating, times = 1, p = 0.1,
                                  list = FALSE)

temp <- get_training_and_test(ratings, test_index)
edx <- temp$train
validation <- temp$test

remove(ratings, test_index, temp)

log_info("Training data created.")
log_info(paste("Training set:", nrow(edx), "observations.",
                "Test set:", nrow(validation), "observations."))

# Add movie data to training and test sets.

log_info("Adding movie data...")

movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.table(movies) %>%
  mutate(movieId = as.numeric(movieId))

# For display purposes, shorten the name of each genre to two characters.
movies <- movies %>%
  mutate(genres = gsub("([^\\|]{2})[^\\|]+", "\\1", genres)) %>%
  mutate(genres = sub("Sc", "SF", genres)) %>%
  mutate(genres = sub("Fi", "FN", genres)) %>%
  mutate(genres = sub("\\(n", "No", genres))

# Separate the movie title and year into separate variables.
title_year <- map_dfr(movies$title, function(t) {
  tt <- substr(t, 1, str_length(t) - 7)
  yr <- substr(t, str_length(t) - 4, str_length(t) - 1)
  tibble(title = trimws(tt), year = as.numeric(yr))
})

movies <- cbind(movies[, 1], title_year, movies[, 3])

# Get individual genres for later use.
all_genres <- unique(unlist(sapply(movies$genres,
                                   function(g) strsplit(g, "\\|"))))

edx <- left_join(edx, movies, by = "movieId")
validation <- left_join(validation, movies, by = "movieId")

# Add a variable for how old the movie was when the rating was done.
fractional_year <- function(ts) year(ts) + month(ts) / 13
edx <- edx %>%
  mutate(age = fractional_year(timestamp) - year)
validation <- validation %>%
  mutate(age = fractional_year(timestamp) - year)

remove(title_year, movies, fractional_year)

log_info("Movie data added to training and test sets.")

# Save checkpoint data (don't save logging objects).
save(file = "ml-10M100K/cp01.RData",
     list = setdiff(ls(), ls(pattern = ".*log")))


## 3. Train with cross-validation ---------------------------------------------
# Try a few non-genre models, using 5-fold cross validation on each one.
#
suppressWarnings(set.seed(10, sample.kind = "Rounding"))
kfold_sets <- createFolds(edx$rating, k = 5)

models <- list(model_reg_movie_user,
               model_user_reg_movie,
               model_reg_movie_user_year,
               model_user_reg_movie_year,
               model_year_reg_movie_user,
               model_user_year_reg_movie,
               model_age_reg_movie_user,
               model_age_user_reg_movie,
               model_user_reg_movie_age,
               model_reg_movie_user_age)

# For each model:
model_results <- map_dfr(models, function(model) {
  log_info(paste(model$name, "- cross validation training"))

  # For each fold:
  tunings <- map_dfr(seq_len(length(kfold_sets)), function(i) {
    log_info(paste("Training on k-fold set", i))

    # Construct the training and test sets.
    temp <- get_training_and_test(edx, kfold_sets[[i]])
    train <- temp$train ; test <- temp$test ; remove(temp)

    # Apply the model with a range of tuning parameters.
    lambdas <- seq(0, 8, 0.5)
    rmses <- sapply(lambdas, function(l) {
      fit <- model$train(train, l)
      predicted <- fit$predict(test)
      rmse(predicted, test$rating)
    })

    # Return the best tuning parameter and its corresponding error.
    min_rmse <- which.min(rmses)
    return(tibble(lambda = lambdas[min_rmse], rmse = rmses[min_rmse]))
  })
  log_info("Training on k-fold sets ended.")

  # Average the tuning parameters on the folds to get a new value
  # and use the model on the entire training set to calculate a new error.
  best_lambda <- mean(tunings$lambda)
  fit <- model$train(edx, best_lambda)
  predicted <- fit$predict(edx)
  rmse <- rmse(predicted, edx$rating)

  log_info(paste("Best fit for", model$name, ": lambda =", best_lambda,
                 "RMSE =", rmse))

  tibble(name = model$name, lambda = best_lambda, RMSE = rmse, fit = fit)
})

remove(kfold_sets)

# Pick the best model.
model_index <- which.min(model_results$RMSE)
chosen_nongenre_model <- model_results[model_index, ]
log_info(paste("The model with the best fit is", chosen_nongenre_model$name,
               "with RMSE", chosen_nongenre_model$RMSE))

nongenre_predictions <- chosen_nongenre_model$fit$predict(edx)
val_nongenre_predictions <- chosen_nongenre_model$fit$predict(validation)

nongenre_results <- model_results %>% 
  arrange(desc(RMSE)) %>%
  select(-fit)
saveRDS(nongenre_results, file = "nongenre_results.rds", ascii = TRUE)

remove(model_results, model_index, models)

# Save checkpoint data (don't save logging objects).
save(file = "ml-10M100K/cp02.RData",
     list = setdiff(ls(), ls(pattern = ".*log")))

## 4. Create genre model ------------------------------------------------------
# Add individual genre variables to the training and validation sets.
#
log_info("Adding individual genre variables to training and validation sets.")
separate_genres <- map_dfc(
  all_genres, function(g) ifelse(grepl(g, edx$genres), 1, 0))
colnames(separate_genres) <- all_genres
edx <- edx %>% select(-title, -year, -timestamp, -age, -genres)
edx <- cbind(edx, separate_genres)

separate_genres <- map_dfc(
  all_genres, function(g) ifelse(grepl(g, validation$genres), 1, 0))
colnames(separate_genres) <- all_genres
validation <- validation %>% select(-title, -year, -timestamp, -age, -genres)
validation <- cbind(validation, separate_genres)

remove(separate_genres)

# Calculate residuals left after non-genre model predictions are removed.
edx_residuals <- edx %>%
  mutate(rating = rating - nongenre_predictions)
edx <- edx %>% select(rating)

# Produce a weighted genre vector for each user in two steps.
# First, spread each rating evenly across a movie's genres.
log_info("Producing a weighted genre vector for each user.")
user_mean_rating <- edx_residuals %>%
  mutate(genre_count = rowSums(across(all_of(all_genres)))) %>%
  mutate(across(all_of(all_genres), function(x) {
    ifelse(genre_count != 0, x * rating / genre_count, 0)
  })) %>%
  select(userId, rating, all_of(all_genres))

# Second, compute the average rating for each genre, counting only the rows
# that contain the genre.
user_mean_rating <- user_mean_rating %>%
  group_by(userId) %>%
  summarize(across(.cols = all_of(all_genres),
                   .fns = c(
                     function(x) ifelse(any(x != 0), sum(x) / sum(x != 0), 0))))
colnames(user_mean_rating) <- c("userId", all_genres)

log_info("Completed weighted genre vectors.")

# The predicted rating for a movie is then the sum of the user's mean
# ratings of each of the movie's genres.
genre_predictfn <- function(dataset) {
  # Calculate in groups of ~10,000 ratings each, since matrix algebra is fast.
  # Credit goes to shan2382 at stackoverflow.com for the following ingenious
  # way of creating 900 groups of indices.
  # https://stackoverflow.com/questions/3318333/split-a-vector-into-chunks
  groups <- split(seq_len(nrow(dataset)), sort(seq_len(nrow(dataset)) %% 900))

  res <- lapply(groups, function(g) {
    if (first(g) %% 500000 < 1000)
      log_info(paste("Processing records up to", last(g)))

    # Get movies to be rated and extract their genres.
    movie_ratings <- dataset[g, ]
    movie_genres <- as.matrix(movie_ratings %>%
                                select(all_of(all_genres)))

    # Find the user mean ratings for the users rating those movies.
    user_idx <- match(movie_ratings$userId, user_mean_rating$userId)
    user_genres <- as.matrix(user_mean_rating[user_idx, ] %>%
                               select(all_of(all_genres)))

    # Multiply to get the predicted ratings. The diagonal of the
    # resulting matrix contains the predictions for the selected users.
    diag(user_genres %*% t(movie_genres))
  })
  unlist(res)
}

# Save checkpoint data (don't save logging objects).
save(file = "ml-10M100K/cp03.RData",
     list = setdiff(ls(), ls(pattern = ".*log")))

## 5. Test genre model -------------------------------------------------------
# Check results of combining non-genre and genre models on the training set.
#
log_info("Checking the result of non-genre and genre models together.")

genre_predictions <- genre_predictfn(edx_residuals)

genre_tuning <- map_dfr(seq(.1, 4, .1), function(x) {
  combined_predictions <- nongenre_predictions + x * genre_predictions
  tibble(lambda = x, rmse = rmse(combined_predictions, edx$rating))
})

tuning_index <- which.min(genre_tuning$rmse)
log_info(paste("Best fit for combined models: lambda =",
               genre_tuning$lambda[tuning_index],
               "RMSE =", genre_tuning$rmse[tuning_index]))

combined_results <- tibble(name = paste(last(nongenre_results$name),
                                        "+ Genre Effects"),
                           lambda = last(nongenre_results$lambda),
                           lambda2 = genre_tuning$lambda[tuning_index],
                           RMSE = genre_tuning$rmse[tuning_index])

remove(edx_residuals)

## 6. Final Check ------------------------------------------------------------
#
# Test the model on the validation set.
#
validation_residuals <- validation %>%
  mutate(rating = rating - val_nongenre_predictions)

# Save checkpoint data (don't save logging objects).
save(file = "ml-10M100K/cp04.RData",
     list = setdiff(ls(), ls(pattern = ".*log")))

validation_predictions <- val_nongenre_predictions +
  genre_tuning$lambda[tuning_index] * genre_predictfn(validation_residuals)

validation_rmse <- rmse(validation_predictions, validation$rating)
log_info(paste("Final RMSE is", validation_rmse))

combined_results <- rbind(combined_results,
                          tibble(name = "", lambda = 0, lambda2 = 0,
                                 RMSE = validation_rmse))
saveRDS(combined_results, file = "combined_results.rds", ascii = TRUE)
