require(tidyverse)
require(caret)
require(data.table)
require(lubridate)

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

edx <- edx %>% mutate(timelag = year(timestamp) - year)
validation <- validation %>% mutate(timelag = year(timestamp) - year)

remove(title_year, movies)

log_info("Movie data added to training and test sets.")

# Save checkpoint data (don't save logging objects).
save(file = 'ml-10M100K/cp01.RData',
     list = setdiff(ls(), ls(pattern = ".*log")))


##############################################################################
# Try a few linear models, using 5-fold cross validation on each one.
#
suppressWarnings(set.seed(10, sample.kind = "Rounding"))
kfold_sets <- createFolds(edx$rating, k = 5)

models <- list(linear_movie_user_model, 
               linear_movie_user_year_model, 
               linear_movie_user_timelag_model,
               linear_movie_user_genres_model)

# For each model:
model_results <- map_dfr(models, function(model) {
  log_info(paste(model$name, "- cross validation training"))

  # For each fold:
  tunings <- map_dfr(seq_len(length(kfold_sets)), function(i) {
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
  
  tibble(name = model$name, lambda = best_lambda, RMSE = rmse, fit = fit)
})

remove(kfold_sets)

# Pick the best model.
model_index <- which.min(model_results$RMSE)
chosen_linear_model <- model_results[model_index, ]
log_info(paste("The model with the best fit is", chosen_linear_model$name,
               "with RMSE", chosen_linear_model$RMSE))

linear_model_predictions <- chosen_linear_model$fit$predict(edx)

remove(model_results, model_index)

# Save checkpoint data (don't save logging objects).
save(file = 'ml-10M100K/cp02.RData',
     list = setdiff(ls(), ls(pattern = ".*log")))

##############################################################################
# Add individual genre variables to the training and validation sets.
#
log_info("Adding individual genre variables to training and validation sets.")
separate_genres <- map_dfc(
  all_genres, function(g) ifelse(grepl(g, edx$genres), 1, 0))
colnames(separate_genres) <- all_genres
edx <- edx %>% select(-title, -year, -timestamp, -genres)
edx <- cbind(edx, separate_genres)

separate_genres <- map_dfc(
  all_genres, function(g) ifelse(grepl(g, validation$genres), 1, 0))
colnames(separate_genres) <- all_genres
validation <- validation %>% select(-title, -year, -timestamp, -genres)
validation <- cbind(validation, separate_genres)

remove(separate_genres)

# Calculate residuals left after linear model predictions are removed.
edx_residuals <- edx %>%
  mutate(rating = rating - linear_model_predictions)

# Produce a weighted genre vector for each user in two steps.
# First, spread each rating evenly across a movie's genres.
log_info("Producing a weighted genre vector for each user.")
user_mean_rating <- edx_residuals %>%
  mutate(genre_count = rowSums(across(all_of(all_genres)))) %>%
  mutate(across(all_of(all_genres), function(x) {
    ifelse(genre_count != 0, x * rating / genre_count, 0)
  })) %>%
  select(userId, rating, all_of(all_genres))

# Second, computer the average rating for each genre.
user_mean_rating <- user_mean_rating %>%
  group_by(userId) %>%
  summarize(across(all_of(all_genres), function(x) mean(x)))

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
    
    # Multiply to get the predicted ratings.
    diag(user_genres %*% t(movie_genres))
  })
  unlist(res)
}

# Save checkpoint data (don't save logging objects).
save(file = 'ml-10M100K/cp03.RData',
     list = setdiff(ls(), ls(pattern = ".*log")))

##############################################################################
# Check results of combining linear and genre models on the training set.
#
log_info("Checking the result of linear and genre models together.")

genre_predictions <- genre_predictfn(edx_residuals)
combined_predictions <- linear_model_predictions + genre_predictions
combined_rmse <- .rmse(combined_predictions, edx$rating)

log_info(paste("Combined models: RMSE =", combined_rmse))

remove(edx_residuals)

##############################################################################
##############################################################################
# FINAL CHECK
#
# Test the model on the validation set.
#
validation_linear_predictions <- chosen_linear_model$fit$predict(validation)
validation_residuals <- validation %>%
  mutate(rating = rating - validation_linear_predictions)

# Save checkpoint data (don't save logging objects).
save(file = 'ml-10M100K/cp04.RData',
     list = setdiff(ls(), ls(pattern = ".*log")))

validation_predictions <- validation_linear_predictions +
  genre_predictfn(validation_residuals)
validation_rmse <- .rmse(validation_predictions, validation$rating)
log_info(paste("Final RMSE is", validation_rmse))
