rm(list = ls())
gc()

# install.packages("devtools")  # if you don't have this installed yet.
# devtools::install_github('susanathey/causalTree')

library(causalTree)

library(grf)
library(rpart)
library(glmnet)
library(splines)
library(MASS)
library(lmtest)
library(sandwich)
library(ggplot2)
library(stringr)

data <- read.csv("D:/R/ML-based casual inference/welfare-small.csv")
head(data)
n <- nrow(data)

treatment <- "w"
outcome <- "y"
covariates <- c("age", "polviews", "income", "educ", "marital", "sex")




# Prepare dataset
fmla <- formula(paste0("~ 0 + ", paste0(covariates, collapse="+")))
X <- model.matrix(fmla, data)
W <- data[,treatment]
Y <- data[,outcome]

# Number of rankings that the predictions will be ranking on
# (e.g., 2 for above/below median estimated CATE, 5 for estimated CATE quintiles, etc.)
num.rankings <- 5  

# Prepare for data.splitting
# Assign a fold number to each observation.
# The argument 'clusters' in the next step will mimick K-fold cross-fitting.
num.folds <- 10
folds <- sort(seq(n) %% num.folds) + 1

# Comment or uncomment depending on your setting.
# Observational setting with unconfoundedness+overlap (unknown assignment probs):
# forest <- causal_forest(X, Y, W, clusters = folds)
# Randomized settings with fixed and known probabilities (here: 0.5).
forest <- causal_forest(X, Y, W, W.hat=.5, clusters = folds)

# Retrieve out-of-bag predictions.
# Predictions for observation in fold k will be computed using
# trees that were not trained using observations for that fold.
tau.hat <- predict(forest)$predictions
hist(tau.hat)

test_calibration(forest)

variable_importance(forest)

