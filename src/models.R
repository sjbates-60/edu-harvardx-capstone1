#-------------------------------------------------------------
# Partitions a dataset into a test set and training set.
# Parameters:
#   dataset    - A set of movie ratings.
#   test_index - An index of rows for the test set.
# Returns:
#   A list of two elements: train and test.
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



.rmse <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}


#---------------------------------------------------------------
# Each of the following objects contains a model used to predict
# movie ratings. Each object is a list containing two elements:
# - a name indicating the model used; and
# - a train() function used to train the model.
# The train() function takes a dataset and a tuning parameter
# and returns a list containing a predict() function to run
# on a dataset.
#
linear_movie_user_model <- list(
  name = "Regularized Movie + User Effects",
  train = function(dataset, lambda) {
    mu <- mean(dataset$rating)
    
    b_i <- dataset %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + lambda))
    
    b_u <- dataset %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
    
    predictfn = function(dataset) {
      predicted_ratings <- dataset %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(pred = mu + b_i + b_u) %>%
        pull(pred)
      return(predicted_ratings)
    }
    return(list(predict = predictfn))
  }
)

linear_movie_user_year_model <- list(
  name = "Regularized Movie + User + Year Effects",
  train = function(dataset, lambda) {
    mu <- mean(dataset$rating)
    
    b_i <- dataset %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + lambda))
    
    b_u <- dataset %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
    
    b_y <- dataset %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      group_by(year) %>%
      summarize(b_y = sum(rating - b_u - b_i - mu) / (n() + lambda))
    
    predictfn = function(dataset) {
      predicted_ratings <- dataset %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_y, by = "year") %>%
        mutate(pred = mu + b_i + b_u + b_y) %>%
        pull(pred)
      return(predicted_ratings)
    }
    return(list(predict = predictfn))
  }
)

linear_movie_user_genres_model <- list(
  name = "Regularized Movie + User + Genres Effects",
  train = function(dataset, lambda) {
    mu <- mean(dataset$rating)
    
    b_i <- dataset %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + lambda))
    
    b_u <- dataset %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
    
    b_g <- dataset %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - b_u - b_i - mu) / (n() + lambda))
    
    predictfn = function(dataset) {
      predicted_ratings <- dataset %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_g, by = "genres") %>%
        mutate(pred = mu + b_i + b_u + b_g) %>%
        pull(pred)
      return(predicted_ratings)
    }
    return(list(predict = predictfn))
  }
)
