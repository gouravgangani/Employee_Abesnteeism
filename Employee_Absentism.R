#Employee Absenteeism

rm(list=ls())

#------------------------------ Data Load --------------------------#
absentData <- read.csv("D:\\Edwisor\\Absentism\\absent.csv")
head(absentData)

#Work.load.average.day has comma in the number
absentData$Work.load.Average.day <- gsub(",","", absentData$Work.load.Average.day)


#-------------------------- Exploratory Analysis -------------------#

#Exploratory Data Analysis
str(absentData)
summary(absentData)

head(absentData)



#----------------------- Data Type Conversion ---------------------#
absentData$ID = as.factor(as.character(absentData$ID))

absentData$Reason.for.absence[absentData$Reason.for.absence %in% 0] = NA
absentData$Reason.for.absence = as.factor(as.character(absentData$Reason.for.absence))

absentData$Month.of.absence[absentData$Month.of.absence %in% 0] = NA
absentData$Month.of.absence = as.factor(as.character(absentData$Month.of.absence))

absentData$Day.of.the.week = as.factor(as.character(absentData$Day.of.the.week))
absentData$Seasons = as.factor(as.character(absentData$Seasons))
absentData$Disciplinary.failure = as.factor(as.character(absentData$Disciplinary.failure))
absentData$Education = as.factor(as.character(absentData$Education))
absentData$Son = as.factor(as.character(absentData$Son))
absentData$Social.drinker = as.factor(as.character(absentData$Social.drinker))
absentData$Social.smoker = as.factor(as.character(absentData$Social.smoker))
absentData$Pet = as.factor(as.character(absentData$Pet))
absentData$Work.load.Average.day = as.numeric(absentData$Work.load.Average.day)
absentData$Body.mass.index = as.numeric(absentData$Body.mass.index)




#------------------------- Missing Value Analysis -----------------#
#Total percentage of missing values in the dataset
library(VIM)
missingValue <- as.data.frame(colSums(is.na(absentData)))
sum(missingValue)
sum(missingValue*100)/nrow(absentData) # 23.10 % of the values are missing from the dataset and we cannot delete such a big amount of missing values 

missingValueData <- kNN(absentData, k = 5)
summary(missingValueData)
newData <- subset(missingValueData, select = ID:Absenteeism.time.in.hours)

#Generating BoxPlots to see outliers
boxplot(newData$Transportation.expense, xlab = "Transportation Expense", ylab = "Values") #Has Outliers
boxplot(newData$Distance.from.Residence.to.Work, xlab = "Distance between Office and Home", ylab = "Miles") #No Outliers
boxplot(newData$Service.time, xlab = "Service Time", ylab = "Time")#Has Outliers
boxplot(newData$Age, xlab = 'Age')#Has Outliers
boxplot(newData$Work.load.Average.day, xlab = "Work Load", ylab = "Days") #Has outliers
boxplot(newData$Hit.target, xlab ="Target") #no outliers
boxplot(newData$Weight, xlab = "Weight") #no outliers
boxplot(newData$Height, xlab = 'Height') #Has outliers
boxplot(newData$Body.mass.index, xlab = "Body Mass Index") # no outliers
boxplot(newData$Absenteeism.time.in.hours, xlab = "Absenteeism", ylab = "Hours")#Has outliers

#Creating Benchmarks to remove outliers
t_Expense    <- 260 + 1.5*IQR(newData$Transportation.expense)
service_Time <- 16 + 1.5*IQR(newData$Service.time)
age          <- 40 + 1.5*IQR(newData$Age)
work_Load    <- 284853 + 1.5*IQR(newData$Work.load.Average.day)
height       <- 172 + 1.5*IQR(newData$Height)
absent_Hours <- 8 + 1.5*IQR(newData$Absenteeism.time.in.hours)

#newData < - newData[newData$Transportation.expense < t_Expense]


cleanData <- subset(newData, Transportation.expense <= t_Expense)
cleanData <- subset(cleanData, Service.time <= service_Time)
cleanData <- subset(cleanData, Work.load.Average.day <= work_Load)
cleanData <- subset(cleanData, Height <= height)
cleanData <- subset(cleanData, Absenteeism.time.in.hours <= absent_Hours)

cleanData <- cleanData[complete.cases(cleanData),]

boxplot(cleanData$Height, xlab = 'Height') 
boxplot(cleanData$Transportation.expense)                  

#bScaling <- sample(2, nrow(cleanData), replace = TRUE, prob = c(0.8,0.2))
#bScaleTrain <- cleanData[bScaling == 1, ]   
#bScaleTest  <- cleanData[bScaling == 2, ] 

#bScaleTest$Reason.for.absence[bScaleTest$Reason.for.absence == 3 ] <- NA
#bScaleTest <- bScaleTest[complete.cases(bScaleTest),]



#-------------------------------- Feature Selection ------------------------#
library(usdm)
library(corrplot)

#Correlation Analysis
numeric_index = sapply(cleanData, is.numeric)
numeric_data = cleanData[,numeric_index]
vifcor(numeric_data)

corrplot(cor(numeric_data), method = "number", addCoef.col="grey", order = "AOE")
pairs(numeric_data)

corPlot <- cor(numeric_data)
corrplot(corPlot, method = "shade")

#-------------------------------- Feature Scaling ----------------------#
#Variability is among the variables. So we need to normalize the data

library("dplyr")
library(faraway)
numericData <- select_if(cleanData, is.numeric)
factorData  <- select_if(cleanData, is.factor)
numericData <- as.data.frame(scale(numericData))

#This will help us to get the predictions on the original scale
#scaleList   <- list(scale = attr(numericData,"scaled:scale"),
#                    center = attr(numericData,"scaled:center"))

finalData   <- data.frame(factorData,numericData)
finalData   <- data.frame(finalData)

#To check the correlations
corTable <- round(cor(numericData),2)
corTable[corTable <= 0.5] <- 0

#---------------------------- Data Partitioning ------------------------#
ind   <- sample(2, nrow(finalData), replace = T, prob = c(0.8,0.2))
train <- finalData[ind == 1,]
test  <- finalData[ind == 2,]

test$Reason.for.absence[test$Reason.for.absence == 17 | test$Reason.for.absence == 3 
                        | test$Reason.for.absence == 5 | test$Reason.for.absence == 4 
                        | test$Reason.for.absence == 24] <- NA
test$Pet[test$Pet == 8] <- NA
test <- test[complete.cases(test),]



#---------------------------- Linear Regression -----------------------#
set.seed(1234)

library(caret)
linearModel <- lm(Absenteeism.time.in.hours~., data=train[,!colnames(train) %in% c("ID")])

summary(linearModel)
predictLinear <- predict(linearModel, test, type="response")



#omod <- lm(Absenteeism.time.in.hours~., data=bScaleTrain[,!colnames(bScaleTrain) %in% c("ID")])
#summary(omod)
#op <- predict(omod, bScaleTest)
#usp <- predictLinear * scaleList$scale['Absenteeism.time.in.hour'] 
#+ scaleList$center["Absenteeism.time.in.hour"]
#all.equal(op, usp)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = linearModel, obs = test$Absenteeism.time.in.hours))


predictedData <- data.frame(test$ID,predictLinear)
#lrResult <- postResample(pred = predictLinear, obs = test$Absenteeism.time.in.hours)


#Model Performance
plot(test$Absenteeism.time.in.hours, type = 'l', lty = 1.8, col = 'green')
lines(predictLinear, type = 'l', col = 'blue')

hist(test$Absenteeism.time.in.hours)
hist(predictLinear)

#New Data Frame containing the Predicted value against the employee ID
write.csv(predictedData,"D:\\Edwisor\\\\Absentism\\test.csv", row.names = TRUE)

#------------------------------- Random Forest ----------------------------#
library(randomForest)
rfModel <- randomForest(Absenteeism.time.in.hours~., data = train, ntree = 500)
summary(rfModel$predicted)
predictRandom <- predict(rfModel, test)
rfResult <- postResample(pred = predictRandom, obs = test$Absenteeism.time.in.hours)


hist(predictRandom)

#------------------------------- Decision Tree ----------------------------#
library(rpart)
decisionModel   <- rpart(Absenteeism.time.in.hours~., data = train, method = "anova")
predictDecision <- predict(decisionModel,test)

#Create data frame for actual and predicted values
dtData  <- data.frame(test$ID,predictDecision)

#Calcuate MAE, RMSE, R-sqaured for testing data 
library(caret)
dResult <- postResample(pred = predictDecision, obs = test$Absenteeism.time.in.hours)

#-------------------------- Model Improvement -------------------#
#1. Found Multicolinearity 
#2. More number of independent variables. 
#3. Poor accuracy in the models. 



#--- Dimensionality Reduction using PCA - Test -------------------#
#Out of 21 Variables we have 11 variables with Factor Datatype, hence we need to convert them to numeric
library(dummies)

newData <- dummy.data.frame(finalData, names = c("ID","Reason.for.absence","Month.of.absence",
                                                 "Disciplinary.failure","Education","Son",
                                                 "Social.drinker","Social.smoker","Pet","Seasons",
                                                 "Day.of.the.week"))

#Added numeric data in the dataframe. Before it was missing the predictor variable.                                                                                                  "Day.of.the.week"))
newData <- data.frame(numericData,newData)

#Data Partinioning 
ind <- sample(2,nrow(newData), replace = T, prob = c(0.7,0.3))
Train <- newData[ind == 1,]
Test  <- newData[ind == 2,]

#Principal component analysis
pComp <- prcomp(Train)

#Standard deviation of each principal component
pStdev <- pComp$sdev

#Variance
pVar <- pStdev^2

#Proportion of variance explained
proVar <- pVar/sum(pVar)

plot(cumsum(proVar), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")


#Add a training set with principal components
#train.data = data.frame(Absenteeism.time.in.hours = Train$Absenteeism.time.in.hours, pComp$x)
pcaTrain <- data.frame(Absenteeism.time.in.hours = Train$Absenteeism.time.in.hours, pComp$x)


# From the above plot selecting 45 components since it explains almost 95+ % data variance
#train.data =train.data[,1:9]
pcaTrain <- pcaTrain[,1:50]


#Transform test data into PCA
#test.data = predict(prin_comp, newdata = Test)
pcaTest <- predict(pComp, newdata = Test)
pcaTest <- as.data.frame(pcaTest)
#test.data = as.data.frame(test.data)

#Select the first 9 components
pcaTest <- pcaTest[,1:50]

#-------------------------- Decision Tree with PCA  ------------------------------------#
#RMSE: 0.2814679 
#Rsquared: 0.915544    
#MAE: 3 0.1724320
  
dtModelImproved <- rpart(Absenteeism.time.in.hours ~., data = pcaTrain, method = "anova")

#Predict the test cases
dtPredictionImproved <- predict(dtModelImproved,pcaTest)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred <- dtPredictionImproved, obs = Test$Absenteeism.time.in.hours))

#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")
lines(dtPredictionImproved,col="blue")

#Create data frame for actual and predicted values
df_pred <- data.frame("actual"=Test$Absenteeism.time.in.hours.1, "dt_pred"= dtPredictionImproved)

#Unscaling the data to get Actual Predictions
Test * attr(Test, 'scaled:scale') + attr(Test, 'scaled:center')

#------------------------------ Random Forest with PCA ------------------------------------------------------#
#RMSE: 0.2546469
#R squared: 0.9445420
#MAE: 0.1460709

#Train the Model 
rfModelImproved <- randomForest(Absenteeism.time.in.hours~., data = pcaTrain, ntrees = 500)

#Prediction on test data
rfPredictionImproved <- predict(rfModelImproved,pcaTest)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = rfPredictionImproved, obs = Test$Absenteeism.time.in.hours))


#Plot a graph for actual vs predicted values
plot(Test$Absenteeism.time.in.hours,type="l",lty=2,col="green")
lines(rfPredictionImproved,col="blue")


#------------------------------ Linear Regression with PCA ------------------------#
#RMSE: 0.02477026
#R squared: 0.99936171
#MAE: 0.01694319

#Train the Model
lrModelImproved <- lm(Absenteeism.time.in.hours ~ ., data = pcaTrain)
summary(lrModelImproved)

#Predict the test cases
lrPredictionsImproved <- predict(lrModelImproved,pcaTest)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = lrPredictionsImproved, obs =Test$Absenteeism.time.in.hours))

#Plot a graph for actual vs predicted values
plot(Test$Absenteeism.time.in.hours,type="l",lty=2,col="green")
lines(lrPredictionsImproved,col="blue")
