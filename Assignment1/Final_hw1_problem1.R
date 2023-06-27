## NAME: Devyani Mardia
## NETID: dm1633


################
## Code for HW 1 problem 1
##
## INSTRUCTIONS: 
##
##
################

n_experiment = 1000

n = 50
p = 5

lambda = 0

beta_true = rep(1, p)

all_betahats = matrix(0, n_experiment, p)

all_errs = rep(0, n_experiment)

for (i in 1:n_experiment){

    X = matrix(rnorm(n*p), n, p)
    noise = 3*rnorm(n)

    Y = X %*% beta_true + noise

    betahat = solve(t(X) %*% X + lambda * diag(p), t(X) %*% Y)

    all_betahats[i, ] = betahat
    all_errs[i] = sum( (betahat - beta_true)^2 )
}


bias = apply(all_betahats, 2, mean) - beta_true
sds = apply(all_betahats, 2, sd)

print(sprintf("Lambda: %.3f", lambda))
print("Bias: ")
print(bias)
print("Root-variance (sd): ")
print(sds)
print("MSE: ")
print(mean(all_errs))

################
#For lambda = 0

# > print(sprintf("Lambda: %.3f", lambda))
# [1] "Lambda: 0.000"
# 
# > print("Bias: ")
# [1] "Bias: "
# 
# > print(bias)
# [1] -0.001893619 -0.031811729  0.009714371 -0.009614595  0.019364662
# 
# > print("Root-variance (sd): ")
# [1] "Root-variance (sd): "
# 
# > print(sds)
# [1] 0.4603652 0.4532723 0.4522921 0.4425577 0.4442822
# 
# > print("MSE: ")
# [1] "MSE: "
# 
# > print(mean(all_errs))
# [1] 1.015766

################

# 1.1 Describe how the samples Yi,Xi’s are generated in this file. 
#What is the distribution and variance of the noise εi?
#Y is generated using the formula X*Beta + noise 
#X is generated using random generation of the normal distribution.
hist(noise)
#We can see that this is a random generation of normal distribution multiplied by 3,
#so the variance and sd of noise is:

mean(noise)
var(noise)

#> var(noise)
#[1] 7.759882
#> sd(noise)
#[1] 2.785656

#Modify λ to vary between 0, 5, 10, 20, 40. 
#What do you observe about the resulting bias, variance, and MSE?
#   
# > print(sprintf("Lambda: %.3f", lambda))
# [1] "Lambda: 5.000"
# > print("Bias: ")
# [1] "Bias: "
# > print(bias)
# [1] -0.07532932 -0.09333868 -0.10849659 -0.09548788 -0.11122316
# > print("Root-variance (sd): ")
# [1] "Root-variance (sd): "
# > print(sds)
# [1] 0.3925877 0.4113028 0.3900148 0.4068567 0.3936579
# > print("MSE: ")
# [1] "MSE: "
# > print(mean(all_errs))
# [1] 0.8427563


# > print(sprintf("Lambda: %.3f", lambda))
# [1] "Lambda: 10.000"
# > print("Bias: ")
# [1] "Bias: "
# > print(bias)
# [1] -0.1814718 -0.1803357 -0.1927853 -0.1694636 -0.1711914
# > print("Root-variance (sd): ")
# [1] "Root-variance (sd): "
# > print(sds)
# [1] 0.3654820 0.3778367 0.3644749 0.3744299 0.3606619
# > print("MSE: ")
# [1] "MSE: "
# > print(mean(all_errs))
# [1] 0.8394186
# 
# > print(sprintf("Lambda: %.3f", lambda))
# [1] "Lambda: 20.000"
# > print("Bias: ")
# [1] "Bias: "
# > print(bias)
# [1] -0.3022074 -0.3253991 -0.3024864 -0.2952783 -0.3188279
# > print("Root-variance (sd): ")
# [1] "Root-variance (sd): "
# > print(sds)
# [1] 0.3225250 0.3132404 0.3209832 0.3254064 0.3095414
# > print("MSE: ")
# [1] "MSE: "
# > print(mean(all_errs))
# [1] 0.9839228


# > print(sprintf("Lambda: %.3f", lambda))
# [1] "Lambda: 40.000"
# > print("Bias: ")
# [1] "Bias: "
# > print(bias)
# [1] -0.4791866 -0.4648940 -0.4548443 -0.4581660 -0.4649786
# > print("Root-variance (sd): ")
# [1] "Root-variance (sd): "
# > print(sds)
# [1] 0.2546062 0.2487600 0.2338815 0.2450773 0.2437106
# > print("MSE: ")
# [1] "MSE: "
# > print(mean(all_errs))
# [1] 1.379314

# bias for lambda : 0, 5, 10, 20, 40: 
# 0:  -0.001893619 -0.031811729  0.009714371 -0.009614595  0.019364662
# 5:  -0.07532932 -0.09333868 -0.10849659 -0.09548788 -0.11122316
# 10: -0.1814718 -0.1803357 -0.1927853 -0.1694636 -0.1711914
# 20: -0.3022074 -0.3253991 -0.3024864 -0.2952783 -0.3188279
# 40: -0.4791866 -0.4648940 -0.4548443 -0.4581660 -0.4649786

# #bias increased as lambda increased
# 
# root-variance for lambda:  0, 5, 10, 20, 40:
#   0:  0.4603652 0.4532723 0.4522921 0.4425577 0.4442822
#   5:  0.3925877 0.4113028 0.3900148 0.4068567 0.3936579
#   10: 0.3654820 0.3778367 0.3644749 0.3744299 0.3606619
#   20: 0.3225250 0.3132404 0.3209832 0.3254064 0.3095414
#   40: 0.2546062 0.2487600 0.2338815 0.2450773 0.2437106

# #root-variance or variance decreased as lambda increased  
# mse for lambda:  0, 5, 10, 20, 40: 
#   0:  1.015766
#   5:  0.8427563
#   10: 0.8394186
#   20: 0.9839228
#   40: 1.379314

#Observations: mse seems to decrease at first but then increase again after lambda = 10,
#indicating that ideal lambda to decrease the mse would be around 10,
#which would introduce the perfect bias for the mse to be minimal