require(tidyverse)
require(caret)
require(data.table)

options(digits = 5)
source("src/logfile.R", echo = FALSE)
source("src/models.R", echo = FALSE)

##############################################################################
##############################################################################
#
# MAIN PROGRAM STARTS HERE
#
##############################################################################
##############################################################################

##############################################################################
# Data loading
log_start()

log_info("Loading data...")

records <- readLines("ml-10M100K/ratings.dat")
ratings <- str_split_fixed(records, "\\::", 4)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- as.data.table(ratings) %>%
  mutate(userId = as.numeric(userId),
         movieId = as.numeric(movieId),
         rating = as.numeric(rating),
         timestamp = as.POSIXct(timestamp,
                                tz = "UTC",
                                origin = "1970-01-01 00:00.00 UTC"))
remove(records)

log_info("Ratings data loaded.")

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

##############################################################################
# Add movie data to training and test sets.

log_info("Adding movie data...")

movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.table(movies) %>%
  mutate(movieId = as.numeric(movieId))

# Space saving move: shorten the name of each genre to two characters.
movies <- movies %>% 
  mutate(genres = gsub("([^\\|]{2})[^\\|]+", "\\1", genres)) %>%
  mutate(genres = sub("Sc", "SF", genres)) %>%
  mutate(genres = sub("Fi", "FN", genres)) %>%
  mutate(genres = sub("\\(n", "()", genres))

# Separate the movie title and year into separate variables.
title_year <- map_dfr(movies$title, function(t) {
  tt <- substr(t, 1, str_length(t) - 7)
  yr <- substr(t, str_length(t) - 4, str_length(t) - 1)
  tibble(title = trimws(tt), year = as.numeric(yr))
})

movies <- cbind(movies[, 1], title_year, movies[, 3])

edx <- left_join(edx, movies, by = "movieId")
validation <- left_join(validation, movies, by = "movieId")

remove(title_year, movies)

log_info("Movie data added to training and test sets.")


##############################################################################
# Try several linear models, using 5-fold cross validation on each one.
#
suppressWarnings(set.seed(10, sample.kind = "Rounding"))
kfold_sets <- createFolds(edx$rating, k = 5)

models <- list(linear_movie_user_model, 
               linear_movie_user_year_model, 
               linear_movie_user_genres_model)

# Table for comparing the results of applying different models.
model_results <- data.frame(name = c(), lambda = c(), RMSE = c(), fit = c())

# For each model:
lapply(models, function(model) {
  log_info(paste(model$name, "- cross validation training"))

  # For each fold:
  tunings <- map_dfr(1:length(kfold_sets), function(i) {
    log_info(paste("Training on k-fold set", i))
    
    # Construct the training and test sets.
    temp <- get_training_and_test(edx, kfold_sets[[i]])
    train <- temp$train ; test <- temp$test ; remove(temp)
    
    # Apply the model with several tuning parameters.
    lambdas <- seq(0, 8, 0.5)
    rmses <- sapply(lambdas, function(l) {
      fit <- model$train(train, l)
      predicted <- fit$predict(test)
      .rmse(predicted, test$rating)
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
  rmse <- .rmse(predicted, edx$rating)
  
  log_info(paste("Best fit for", model$name, ": lambda =", best_lambda,
                 "RMSE =", rmse))
  
  # Save the model's results in a table.
  model_results <<- rbind(
    model_results,
    tibble(name = model$name, lambda = best_lambda, RMSE = rmse, fit = fit)
  )
  save(file = "model_results.Rdata", list = c("model_results"))
})

remove(kfold_sets)


##############################################################################
# Pick the best model.
model_index <- which.min(model_results$RMSE)
log_info(paste("The model with the best fit is", model_results$name[model_index]))

# edx <- edx %>%
#   mutate(rating = rating - fit$predict(edx))

# Test the model on the validation set.
# fit <- model_results$fit[model_index]
# predicted <- fit$predict(validation)
# rmse <- .rmse(predicted, validation$rating)
# log_info(paste("Final RMSE is", rmse))

