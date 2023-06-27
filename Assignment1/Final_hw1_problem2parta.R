## NAME:Devyani Mardia
## NETID: dm1633


################
## Code for HW 1 problem 2
##
## INSTRUCTIONS:
## The following file performs ridge regression with cross-validation
##
## For Part (a): Fill in the code in parts labeled "FILL IN".
## For Part (b): Construct the features in the section labeled "part (b)"
##
################

#Complete the code in hw1_problem2 R or Python file by filling in all the parts with the label FILL IN.
#What is the final test error? What is the lambda chosen by CV?


set.seed(1)
movies = read.csv("movies_hw1.csv")

#####
## For part (b): construct your features here.

#I have separated it into another file for convenience

###

#Part (a)

n = 300
test_ix = sample(nrow(movies), nrow(movies) - n)

## Exclude title and vote_average
X = as.matrix(movies[, !(names(movies) %in% c("vote_average", "title", "TV.Movie"))])
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

Y = movies[, "vote_average"]

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

#We can also use a matrix to capture the errors for each K with each lambda
#and then calculate sum/mean of the error for each lambda over all the partitions
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
    #Equation: βˆλ = (X⊤X + λIp)−1X⊤Y
    
    ## FILL IN: compute errors
    #Adding the MSE across the partitions for each lambda and storing individually in errs.
    errs[il] = errs[il] + mean((Xvalid %*% beta_ridge - Yvalid)^2)
    
  }
}

#Picking the lambda_star as the lambda which has the minimum error in the cross validation errors list
lambda_star = lambda_ls[which.min(errs)]
beta_ridge = solve(t(X1) %*% X1 + lambda_star*diag(c(rep(1, p-1), 0)), t(X1) %*% Y1)

beta_ridge_final = beta_ridge
test_error = mean((X2 %*% beta_ridge_final - Y2)^2)
baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   lambda_star: %f", test_error, baseline, lambda_star))

#Response: [1] "Test error: 0.424  Baseline: 0.752   lambda_star: 10.000000"
