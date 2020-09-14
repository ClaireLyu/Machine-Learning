library(readxl)
library(glmnet)
library(caret)
library(tidyverse)
library(randomForest)
library(faraway)
library(spm)
library(neuralnet)
library(ggplot2)

#Load the raw data
concrete1 <- read_excel("concrete.xlsx", skip = 1)
concrete <- concrete1[ , c(-1, -2, -4, -5, -11, -12, -13, -14, -16, -20, -21, -22, -23, -24, -25, -26, -27, -29, -33, -34)]
concrete[, c(-4,-5)] <- sapply(concrete[, c(-4,-5)], scale)
concrete[,1]<-sapply(concrete[,1], log)
concrete$Plant <- as.factor(concrete$Plant)
concrete <- as.data.frame(concrete)
View(concrete)

#Load the clean data
concrete <- read_csv("C:/Users/jesso/Downloads/clean_data_28days.csv")
colnames(concrete) <- c("AEA", "AWRA", "W", "Output7", "Output28", "Coarse", "Fine", "Ash")
order_name <- c("AEA", "AWRA", "W", "Coarse", "Fine", "Ash", "Output28", "Output7")
concrete <- concrete[, order_name]
mu <- mean(concrete$Output28)
sd <- sd(concrete$Output28)
concrete <- as_tibble(apply(concrete, 2, scale))
concrete <- concrete[order(concrete$Output28), ]



#Data Visualization
plot(concrete$`%Fly Ash`, main='scatter plot')
concrete %>%
  ggplot() + 
  geom_point(mapping = aes(x = `%Fly Ash`, y = `28 Day`))

#Data Transformation
concrete[, c(1,2,8)] <- concrete[, c(1,2,8)] + 0.01
concrete[, c(3,5,6,7)] <- log(concrete[, c(3,5,6,7)] + 0.01)
concrete[,5]<-sapply(concrete[,5], log)

#Data splitting
set.seed(200)
training_samples <- concrete$Output28 %>%
  createDataPartition(p = 0.8, list = FALSE)

x <- model.matrix(Output28 ~ . - Output7 - 1, data = concrete)
X <- poly(x, degree = 4, raw = TRUE)
X <- cbind(X, concrete$Output28)
X <- as.data.frame(X)
y <- as.matrix(select(concrete, Output28))

# low strength X and y
x_train <- X[c(1:3720), ]
y_train <- y[c(1:3720)]
# high strength X and y
x_test <- X[-c(1:3720), ]
y_test <- y[-c(1:3720)]

# training X and y
x_train <- x[training_samples[, 1], ]
y_train <- y[training_samples[, 1]]
# testing X and y
x_test <- x[-training_samples[, 1], ]
y_test <- y[-training_samples[, 1]]
is.list(as.matrix(x_train))

############################LS
lmod <- lm(V210 ~ ., data = X[training_samples[, 1], ])
summary(lmod)
pred <- predict(lmod, X[-training_samples[, 1], ])

##########################LASSO
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, type.measure = "mse", nfold = 5)
pred <- predict(cv_lasso, x_test, s = "lambda.min")

#####################elastic net
cv_en <- cv.glmnet(x_train, y_train, alpha = 0.5, type.measure = "mse", nfold = 5)
pred <- predict(cv_en, x_test, s = "lambda.min")

plot(pred, (yy*sd + mu))
coef(cv_lasso, cv_lasso$lambda.1se)
cv_lasso$lambda.1se
plot(cv_lasso, xvar = "lambda", label = TRUE)

#################random forest
cv.rf <- RFcv(x_train, y_train, cv.fold = 5, ntree = 100)
#cv.rf <- rfcv(x_train, y_train, cv.fold = 5)
rf <- randomForest(x_train, y_train)
pred = predict(rf, newdata = x_test)

View(tibble(pred*sd + mu, y_test*sd + mu))
sum(pred-mean(y_test))^2/sum(y_test - mean(y_test))^2
######################ANN
n <- neuralnet(Output28 ~ AEA + AWRA + W + Coarse + Fine + Ash, data = concrete[training_samples[, 1], ], 
               hidden = 2, err.fct = 'sse', stepmax=1e5)
#n <- neuralnet(Output28 ~ AEA + AWRA + W + Coarse + Fine + Ash, data = concrete[c(1:3720), ], hidden = 15, 
#              err.fct = 'sse', algorithm="backprop", learningrate = 0.0001, stepmax=1e4)
plot(n)
View(concrete[training_samples[, 1], ])
pred <- predict(n, concrete[-training_samples[, 1], c(1:6)])

#performance
1 - sum(((pred*sd + mu) - (y_test*sd + mu)) ^ 2)/sum(((y_test*sd + mu) - mean(y_test*sd + mu)) ^ 2)
mean(abs((pred*sd + mu) - (y_test*sd + mu))/(pred*sd + mu))
sqrt(mean(((pred*sd + mu)*0.00689 - (y_test*sd + mu)*0.00689)^2))

#Define a **sequential model** (a linear stack of layers) with 2 fully-connected hidden layers (256 and 128 neurons):
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(6)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)

#Compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#Fit the model by using the train data:
history <- model %>% fit(
  xtrain, ytrain, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

#Evaluate model performance on the test data:
model %>% evaluate(xtest, ytest)

#Potential outliers
halfnorm(hatvalues(mod)) # 4485 and 4962
halfnorm(rstudent(mod)) #7038 and 1669





