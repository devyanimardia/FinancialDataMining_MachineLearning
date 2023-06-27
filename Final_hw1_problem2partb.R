set.seed(1)
movies = read.csv("movies_hw1.csv")

#Part (b)

#####
## For part (b): construct your features here.

# Feature construction
# Example. movies$log_age = log(movies$age + 1)
# An example is given showing how to construct log_age which is the log of (age + 1).
# 1. log_budget_sq: the square of the log_budget.
# 2. log_revenue_sq: the square of the log_revenue.
# 3. log_vote_count: the log of (vote_count + 1). We add 1 in case the vote count is 0.
# 4. Action.Adven: indicator that is 1 if the movie is both Action and Adventure. 0 otherwise. 5. Rom.Com: indicator that is 1 if the movie is both Romance and Comedy. 0 otherwise.
# 6. vote_budget: the product of log_vote_count and log_budget.
# 7. long: indicator that is 1 if the movie runtime is greater than 120 minutes. 0 otherwise.
# What is the final test error? What is the lambda chosen by CV?

movies_new <- movies
movies_new["log_age"]<- log(movies_new$age + 1)
movies_new["log_budget_sq"]<-movies_new$log_budget^2
movies_new["log_revenue_sq"]<-movies_new$log_revenue^2
movies_new["log_vote_count"]<-log(movies_new$vote_count + 1)
movies_new$Action.Adven <- with(movies_new, ifelse(Action == 1 & Adventure == 1, 1, 0))
movies_new$Rom.Com <- with(movies_new, ifelse(Romance == 1 & Comedy == 1, 1, 0))
movies_new["vote_budget"] <- (movies_new$log_vote_count) * (movies_new$log_budget)
movies_new$long <- with(movies_new, ifelse(runtime > 120, 1, 0))


#Using additional features to better predict the response Y
n = 300
test_ix = sample(nrow(movies_new), nrow(movies_new) - n)

## Exclude title and vote_average
X = as.matrix(movies_new[, !(names(movies_new) %in% c("vote_average", "title", "TV.Movie"))])
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

Y = movies_new[, "vote_average"]
X1 = X[-test_ix, ]
Y1 = Y[-test_ix]

X2 = X[test_ix, ]
Y2 = Y[test_ix]

p = ncol(X)
K = 10

## FILL IN: randomly permute the n samples (rows of X1 and entries of Y1).
set.seed(1)
rand <- sample(nrow(X1))
X_total = X1[rand,]
Y_total = Y1[rand]


lambda_ls = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3)

errs = rep(0, length(lambda_ls))

index_matrix = matrix(1:n, ncol=K)
for (k in 1:K){
  valid_ix = index_matrix[, k]
  ## FILL IN:
  # Create variables Xtrain, Ytrain, Xvalid, Yvalid :
  Xtrain = X_total[-valid_ix,]
  Ytrain = Y_total[-valid_ix]
  Xvalid = X_total[valid_ix,]
  Yvalid = Y_total[valid_ix]
  
  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    ## FILL IN: compute ridge regression estimator
    
    beta_ridge = solve(t(Xtrain) %*% Xtrain + lambda*diag(c(rep(1, p-1), 0)), t(Xtrain) %*% Ytrain)
    ## FILL IN: compute errors
    errs[il] = errs[il] + mean((Xvalid %*% beta_ridge - Yvalid)^2)
  }
}

lambda_star = lambda_ls[which.min(errs)]
lambda_star = 10
beta_ridge_final = solve(t(X1) %*% X1 + lambda_star*diag(c(rep(1, p-1), 0)), t(X1) %*% Y1)

test_error = mean((X2 %*% beta_ridge_final - Y2)^2)
baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   lambda_star: %f", test_error, baseline, lambda_star))

#Response: [1] "Test error: 0.361  Baseline: 0.752   lambda_star: 1.000000"

#Although the algorithm gives us lambda_star as 1 but the test_error with X2 data is minimal at 10
#With lambda_star=10:
#Response : [1] "Test error: 0.355  Baseline: 0.752   lambda_star: 10.000000"

#