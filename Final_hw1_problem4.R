## NAME:Devyani Mardia
## NETID:dm1633

################
## Code for HW 1 problem 4
##
## INSTRUCTIONS:
## The following file performs refitted lasso regression with cross-validation
##
## Fill in the code in parts labeled "FILL IN".
##
################


#############
## Lasso optimization algorithm
##
## nothing needs to be done here

softThresh <- function(u, lambda){
  u[abs(u) <= lambda] = 0
  u[u > lambda] = u[u > lambda] - lambda
  u[u < -lambda] = u[u < -lambda] + lambda
  return(u)
}

lassoISTA <- function(X, Y, lambda){
  THRESH = 1e-7
  p = ncol(X)
  n = nrow(X)
  L = 2*eigen(t(X) %*% X)$values[1]

  XtX = t(X) %*% X
  XtY = t(X) %*% Y

  beta = rep(0, p)
  while (TRUE){
    beta_new = softThresh( beta - (2/L)*(XtX %*% beta - XtY), lambda/L)
    if (sum((beta_new - beta)^2) < THRESH)
      break
    else
      beta = beta_new
  }
  return(beta_new)
}

#######
## Problem starts here:
##

cars = read.csv("cars.csv", as.is=TRUE)
set.seed(1)

Y = as.vector(cars[, "mpg"])
X = as.matrix(cars[, !(names(cars) %in% c("mpg", "name"))])
oldX = scale(X)
old_p = ncol(oldX)

## We create interaction features of the form
## column j * column j' for all (j, j')
## column j * (column j')^2 for all (j, j')
## (column j)^2 * (column j')^2 for all (j, j')
for (j in 1:old_p){
  X = cbind(X, oldX*oldX[, j], oldX*oldX[, j]^2, oldX^2*oldX[, j], oldX^2 * oldX[,j]^2)
}

Y = scale(Y)
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

n = 200
test_ix = sample(nrow(X), nrow(X) - n)
X1 = X[-test_ix, ]
X2 = X[test_ix, ]
Y1 = Y[-test_ix]
Y2 = Y[test_ix]

p = ncol(X1)
n = nrow(X1)

K = 5

## FILL IN: randomly permute the n samples (rows of X1 and entries of Y1).
set.seed(1)
rand <- sample(nrow(X1))
X_total = X1[rand,]
Y_total = Y1[rand]


lambda_ls = 10^(seq(-2, 1, 0.1))

errs = matrix(0, length(lambda_ls), K)

index_matrix = matrix(1:n, ncol=K)

for (k in 1:K){
  valid_ix = index_matrix[, k]
  ## FILL IN: create variables Xtrain, Ytrain, Xvalid, Yvalid
  Xtrain = X_total[-valid_ix,]
  Ytrain = Y_total[-valid_ix]
  Xvalid = X_total[valid_ix,]
  Yvalid = Y_total[valid_ix]
  

  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    ## FILL IN: compute lasso estimate with lassoISTA
    beta_lasso = lassoISTA(Xtrain, Ytrain, lambda)

    S = which(abs(beta_lasso) > 1e-10)
    if (length(S) == 0)
      errs[il] = Inf
    else {
      XS = Xtrain[, S]
      ## For refitting, we use ridge regression with a small penalty instead of
      ## OLS in the event that the columns of X are not linearly independent
      #Here I have taken the value of lambda to be lambda but below I am also showing the working with 1e-10 as was given in the document
      
      beta_refit = solve(t(XS) %*% XS + lambda * diag(length(S)), t(XS) %*% Ytrain)
      #Reduce the number of parameters or select only S variables based on the output of beta_lasso
      XvalidS = Xvalid[,S]
      ## FILL IN: compute error 
      errs[il, k] = mean((XvalidS %*% beta_refit - Yvalid)^2)
    }
  }
}
err = rep(0, length(lambda_ls))

for (il in 1:length(lambda_ls)){
  err[il] <- mean(errs[il,])
}

## FILL IN: compute lambda_star 
#take avg in the K partitions and then pick minimum error of the list of lambdas, i.e lambda_ls
lambda_star = lambda_ls[which.min(err)]
beta_lasso = lassoISTA(X1, Y1, lambda_star)

S = which(abs(beta_lasso) > 1e-10)

## FILL IN: compute the refitting on X1, Y1
beta_refit = solve(t(X1[, S]) %*% X1[, S] + lambda_star * diag(length(S)), t(X1[, S]) %*% Y1)

#
beta_ridge_final = beta_refit
## FILL IN: compute the test error
test_error = mean((X2[,S] %*% beta_ridge_final - Y2)^2)


## For comparison, we also compute the OLS
beta_ols = solve(t(X1) %*% X1 + 1e-10 * diag(ncol(X1)), t(X1) %*% Y1)
ols_error = mean((X2 %*% beta_ols - Y2)^2)

## We compute OLS where we only use the first 7 variables and
## the all 1 constant feature.
S = c(1:7, ncol(X1))
beta_ols2 = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)
ols2_error = mean((X2[, S] %*% beta_ols2 - Y2)^2)

baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   OLS: %.3f   OLS (with first 7 vars): %.3f",
              test_error, baseline, ols_error, ols2_error))

#Response: #[1] "Test error: 0.189  Baseline: 1.112   OLS: 0.568   OLS (with first 7 vars): 0.223"


#Now with 1e-10

for (k in 1:K){
  valid_ix = index_matrix[, k]
  ## FILL IN: create variables Xtrain, Ytrain, Xvalid, Yvalid
  Xtrain = X_total[-valid_ix,]
  Ytrain = Y_total[-valid_ix]
  Xvalid = X_total[valid_ix,]
  Yvalid = Y_total[valid_ix]
  
  
  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    ## FILL IN: compute lasso estimate with lassoISTA
    beta_lasso = lassoISTA(Xtrain, Ytrain, lambda)
    
    S = which(abs(beta_lasso) > 1e-10)
    if (length(S) == 0)
      errs[il] = Inf
    else {
      XS = Xtrain[, S]
      ## For refitting, we use ridge regression with a small penalty instead of
      ## OLS in the event that the columns of X are not linearly independent
      #Here I have taken the value of lambda to be lambda but below I am also showing the working with 1e-10 as was given in the document
      beta_refit = solve(t(XS) %*% XS + 1e-10 * diag(length(S)), t(XS) %*% Ytrain)
      #Reduce the number of parameters or select only S variables based on the output of beta_lasso
      XvalidS = Xvalid[,S]
      ## FILL IN: compute error 
      errs[il, k] = mean((XvalidS %*% beta_refit - Yvalid)^2)
    }
  }
}
err = rep(0, length(lambda_ls))

for (il in 1:length(lambda_ls)){
  err[il] <- mean(errs[il,])
}

## FILL IN: compute lambda_star 
#take avg in the K partitions and then pick minimum error of the list of lambdas, i.e lambda_ls
lambda_star = lambda_ls[which.min(err)]
beta_lasso = lassoISTA(X1, Y1, lambda_star)

S = which(abs(beta_lasso) > 1e-10)

## FILL IN: compute the refitting on X1, Y1
beta_refit = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)

beta_ridge_final = beta_refit
## FILL IN: compute the test error
test_error = mean((X2[,S] %*% beta_ridge_final - Y2)^2)


## For comparison, we also compute the OLS
beta_ols = solve(t(X1) %*% X1 + 1e-10 * diag(ncol(X1)), t(X1) %*% Y1)
ols_error = mean((X2 %*% beta_ols - Y2)^2)

## We compute OLS where we only use the first 7 variables and
## the all 1 constant feature.
S = c(1:7, ncol(X1))
beta_ols2 = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)
ols2_error = mean((X2[, S] %*% beta_ols2 - Y2)^2)

baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   OLS: %.3f   OLS (with first 7 vars): %.3f",
              test_error, baseline, ols_error, ols2_error))

#Response: [1] "Test error: 0.145  Baseline: 1.112   OLS: 0.568   OLS (with first 7 vars): 0.223"


#Response for 4.2: We have used beta_lasso for variable selection as
#it enables to push the beta of the variables that are not very relevant to 0 and it doesnt have a closed form
#on the other hand beta_ridge has a proper closed form which enables us to predict beta and in turn 
#the variable Y, this enables us to calculate exactly the test_error 
#by using mean squared error for Y using beta_refit or beta_ridge_final variable
