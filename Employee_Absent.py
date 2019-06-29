# -*- coding: utf-8 -*-

#-------------------------------- Employee Absenteeism Project -------------------------------------#


#Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from fancyimpute import KNN
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline

#-------------------------- Data Load ------------------------------------#
absentData = pd.read_excel("~/Desktop/BGood/Absenteeism_at_work.xls",header=0, delim_whitespace=True)

#----------------------- Exploratory Data Analysis -----------------------#

absentData.shape
absentData.head()
absentData.dtypes

#------------------------ Data Type Conversion ---------------------------#
absentData['ID'] = absentData['ID'].astype('category')
absentData['Reason for absence'] = absentData['Reason for absence'].replace(0,np.nan)
absentData['Reason for absence'] = absentData['Reason for absence'].astype('category')

absentData['Month of absence']   = absentData['Month of absence'].replace(0,np.nan)
absentData['Month of absence']   = absentData['Month of absence'].astype('category')

absentData['Day of the week']    = absentData['Day of the week'].astype('category')
absentData['Seasons']            = absentData['Seasons'].astype('category')
absentData['Disciplinary failure']    = absentData['Disciplinary failure'].astype('category')

absentData['Education']    = absentData['Education'].astype('category')
absentData['Son']          = absentData['Son'].astype('category')
absentData['Social drinker']    = absentData['Social drinker'].astype('category')
absentData['Social smoker']     = absentData['Social smoker'].astype('category')
absentData['Pet']               = absentData['Pet'].astype('category')

absentData.dtypes


#--------------------------------- Missing Value Analysis ---------------------------------#
missingVal  = pd.DataFrame(absentData.isnull().sum()).sum()
missingValPercent = missingVal/len(absentData.index)*100
missingValPercent.round()

#Approx 24% values are null in the dataset. So we need to impute them by suitable method. 

#Missing Value Imputation

absentData.isnull().sum()
absentData = pd.DataFrame(KNN(k = 3).fit_transform(absentData), columns = absentData.columns)
absentData = absentData.round()

#------------------------------- Outlier Analysis -----------------------------------------#
sns.boxplot(data=absentData[['Absenteeism time in hours','Service time','Height','Weight','Transportation expense','Age']])
fig=plt.gcf()
fig.set_size_inches(8,12)
sns.boxplot(data=absentData[['Work load Average/day']])


#Computing the benchmark for the numeric values
numericValues = ['Work load Average/day','Distance from Residence to Work', 'Service time', 'Age','Transportation expense','Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

for i in numericValues:
    q75, q25 = np.percentile(absentData[i], [75,25])
    
    # Computing IQR
    iqr = q75 - q25
    
    # Computing upper and lower threshold/benchmark
    minThres = q25 - (iqr*1.5)
    maxThres = q75 + (iqr*1.5)
    print (maxThres)
    # Replacing all the outliers with NA
    absentData.loc[absentData[i]< minThres,i] = np.nan
    absentData.loc[absentData[i]> maxThres,i] = np.nan


#Impute missing values with KNN
cleanData = pd.DataFrame(KNN(k = 3).fit_transform(absentData), columns = absentData.columns)

# Checking if there is any missing value
cleanData.isnull().sum()
cleanData = cleanData.round()
dummy = cleanData
#-------------------------------------- Feature Selection --------------------------------------#
#Get dataframe with all continuous variables
dfNumeric = cleanData.loc[:,numericValues]

#Check for multicollinearity using corelation graph
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 10))

#Generate correlation matrix
corr = dfNumeric.corr()

#-------------------------------- Feature Scaling ---------------------------------------------#
# Normalization of continuous variables
for i in numericValues:
    if i == 'Absenteeism time in hours':
        continue
    cleanData[i] = (cleanData[i] - cleanData[i].min())/(cleanData[i].max()-cleanData[i].min())
   
#------------------------------- Machine Learning Models --------------------------------------#
#Splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cleanData.iloc[:, cleanData.columns != 'Absenteeism time in hours'], cleanData.iloc[:, 20], test_size = 0.30, random_state = 1)



#------------------------------------ Linear Regression Model ---------------------------------#
# Root Mean Squared Error: 2.898405340060082
# R^2 Score(coefficient of determination) = 0.2772050386036977

from sklearn.linear_model import LinearRegression

#Build Linear regression model
lrModel = LinearRegression().fit(X_train , y_train)

#Perdict for test records
lrModelPred = lrModel.predict(X_test)

#Storing results in a data frame for Actual and Predicted values
lrResult = pd.DataFrame({'Actual': y_test, 'Predicted': lrModelPred})
print(lrResult.head())

#Calculate RMSE and R-squared value
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

print("Root Mean Squared Error: "+str(RMSE(y_test, lrModelPred)))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, lrModelPred)))


#------------------------------------- Decision Tree Model -----------------------------------------#
# Root Mean Squared Error: 4.008437047973632
# R^2 Score(coefficient of determination) = -0.38244228432563765

from sklearn.tree import DecisionTreeRegressor
dtModel = DecisionTreeRegressor(random_state = 1).fit(X_train,y_train)

#Perdict for test records
dtModelPred = dtModel.predict(X_test)

#Storing results in a data frame for Actual and Predicted values
dtResult = pd.DataFrame({'Actual': y_test, 'Predicted': dtModelPred})
print(dtResult.head())

#Define function to calculate RMSE
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test, dtModelPred)))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, dtModelPred)))
    

#---------------------------------- Random Forest Model ---------------------------------------#
# Root Mean Squared Error: 2.737971200171947
# R^2 Score(coefficient of determination) = 0.3550075584440454

from sklearn.ensemble import RandomForestRegressor

#Build random forest 
rfModel = RandomForestRegressor(n_estimators = 500, random_state = 1).fit(X_train,y_train)

#Perdict for test records
rfModelPred = rfModel.predict(X_test)

#Storing results in a data frame for Actual and Predicted values
rfResult = pd.DataFrame({'Actual': y_test, 'Pred': rfModelPred})
print(rfResult.head())

#Calculate RMSE and R-squared value
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

print("Root Mean Squared Error: "+str(RMSE(y_test, rfModelPred )))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, rfModelPred )))
    
  
#------------------------------- Dimensionality Reduction Using PCA --------------------------#
#Store the Dependent Variable 
depVar = cleanData['Absenteeism time in hours']
factorVariable = ['ID','Reason for absence','Month of absence','Day of the week','Seasons','Disciplinary failure', 'Education', 'Social drinker','Social smoker', 'Pet', 'Son']

#PCA works on numeric variables hence converting factor into numeric
pcaData = pd.get_dummies(data = cleanData, columns = factorVariable)

#Import library for PCA
from sklearn.decomposition import PCA
#Converting data to numpy array
nArray = pcaData.values

#Dataset has 116 variables
pca = PCA(n_components=116)
pca.fit(nArray)

#Proportion of variance explained
var = pca.explained_variance_ratio_

#Cumulative screen plot
var1 = np.cumsum(np.round(pca.explained_variance_ratio_ , decimals=4)*100)

#Draw the plot
plt.plot(var1)
plt.show()

pca = PCA(n_components=45)

#Fitting the selected components to the data
pca.fit(nArray)

#Splitting data into train and test data
X_trainImp, X_testImp, y_trainImp, y_testImp = train_test_split(nArray,depVar, test_size=0.3, random_state = 1)

#----------------------- Linear Model Improved Using PCA ----------------------------------------#
#Root Mean Squared Error: 8.633839188446864e-12
#R^2 Score(coefficient of determination) = 1.0

lrModelImproved = LinearRegression().fit(X_trainImp , y_trainImp)

#Perdict for test records
lrModelPredImproved = lrModelImproved.predict(X_testImp)

#Storing results in a data frame for Actual and Predicted values
lrResultImproved = pd.DataFrame({'Actual': y_test, 'Predicted': lrModelPredImproved})
print(lrResultImproved.head())

def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

print("Root Mean Squared Error: "+str(RMSE(y_test, lrModelPredImproved)))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, lrModelPredImproved)))
 
#----------------------- Decision Tree Model Improved Using PCA ----------------------------------------#
#Root Mean Squared Error: 0.0
#R^2 Score(coefficient of determination) = 1.0

dtModelImproved = DecisionTreeRegressor(random_state = 1).fit(X_trainImp,y_trainImp)

#Perdict for test records
dtModelPredImproved = dtModelImproved.predict(X_testImp)

#Storing results in a data frame for Actual and Predicted values
dtResultImproved = pd.DataFrame({'Actual': y_test, 'Predicted': dtModelPredImproved})
print(dtResultImproved.head())

#Define function to calculate RMSE
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test, dtModelPredImproved)))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, dtModelPredImproved)))

#----------------------- Random Forest Model Improved Using PCA ----------------------------------------#
#Root Mean Squared Error: 0.03548759458415528
#R^2 Score(coefficient of determination) = 0.9998916447395986

#Build random forest 
rfModelImproved = RandomForestRegressor(n_estimators = 500, random_state = 1).fit(X_trainImp,y_trainImp)

#Perdict for test records
rfModelPredImproved = rfModelImproved.predict(X_testImp)

#Storing results in a data frame for Actual and Predicted values
rfResultImproved = pd.DataFrame({'Actual': y_test, 'Pred': rfModelPredImproved})
print(rfResult.head())

#Calculate RMSE and R-squared value
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    return rmse

print("Root Mean Squared Error: "+str(RMSE(y_test, rfModelPredImproved)))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test, rfModelPredImproved)))
    
  



