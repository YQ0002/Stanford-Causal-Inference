# https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/policy-evaluation-i---binary-treatment.html#via-sample-means
rm(list = ls())


# use e.g., install.packages("grf") to install any of the following packages.
library(grf)
library(glmnet)
library(splines)
library(ggplot2)

# 5.1 Via sample means

n <- 1000
p <- 4
X <- matrix(runif(n*p), n, p)
W <- rbinom(n, prob = .5, size = 1)  # independent from X and Y
Y <- .5*(X[,1] - .5) + (X[,2] - .5)*W + .1 * rnorm(n) 

y.norm <- 1-(Y - min(Y))/(max(Y)-min(Y)) # just for plotting
plot(X[,1], X[,2], pch=ifelse(W, 21, 23), cex=1.5, bg=gray(y.norm), xlab="X1", ylab="X2", las=1)


par(mfrow=c(1,2))
for (w in c(0, 1)) {
  plot(X[W==w,1], X[W==w,2], pch=ifelse(W[W==w], 21, 23), cex=1.5,
       bg=gray(y.norm[W==w]), main=ifelse(w, "Treated", "Untreated"),
       xlab="X1", ylab="X2", las=1)
}


col2 <- rgb(0.250980, 0.690196, 0.650980, .35)
col1 <- rgb(0.9960938, 0.7539062, 0.0273438, .35)
plot(X[,1], X[,2], pch=ifelse(W, 21, 23), cex=1.5, bg=gray(y.norm), xlab="X1", ylab="X2", las=1)
rect(-.1, -.1, .5, 1.1, density = 250, angle = 45, col = col1, border = NA)
rect(.5, -.1, 1.1, .5, density = 250, angle = 45, col = col1, border = NA)
rect(.5, .5, 1.1, 1.1, density = 250, angle = 45, col = col2, border = NA)
text(.75, .75, labels = "TREAT (A)", cex = 1.8)
text(.25, .25, labels = expression(DO ~ NOT ~ TREAT ~ (A^C)), cex = 1.8, adj = .25)


# Only valid in randomized setting.
A <- (X[,1] > .5) & (X[,2] > .5)
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))



