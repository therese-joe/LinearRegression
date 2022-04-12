# LinearRegression
population of US Cities
Description
The bigcity data frame has 49 rows and 2 columns. The measurements are the population (in 1000's) of 49 U.S. cities in 1920 and 1930. The 49 cities are a random sample taken from the 196 largest cities in 1920.
# Simple Linear Regression

# Dataset

Population of U.S. Cities

# Description

The bigcity data frame has 49 rows and 2 columns.
The measurements are the population (in 1000's) of 49 U.S. cities in 1920 and 1930. The 49 cities are a random sample taken
from the 196 largest cities in 1920.


# Exploring and Understanding Data (EDA)

# 1. Load required Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Collect and load data

df = pd.read_csv('bigcity.csv')
df.tail()

df.drop('Unnamed: 0',axis=1,inplace =True)

df.head()

# 3. Explore numeric variables - five number summary

df.describe()

# 3a. Visualize numeric variables in boxplot and histograms
df.hist()
plt.show()

df.boxplot()

# 3b. Measure spread – variance and standard deviation
# variance and std

df.var()

df.std()

# 4. Explore relationships between variables using scatterplots and two-way cross tabulations

sns.scatterplot(df['u'],df['x'])

pd.crosstab(df['x'],df['u'])

# 5. Transform the dataset
Find the number of rows in given dataset and separate the input and target variables into X and Y. Hint: You can shape function 
to get the size of the dataframe

df.shape

import statsmodels.api as sm
x = df.drop('x',axis=1)
y = df['x']
x = sm.add_constant(x)

# 6. Check the dataset for any missing values and also print out the correlation matrix
You can use .isna() and .corr() functions to check NA's and correlation in the dataframe respectively

#missing values
df.isna().sum()

#correlation matrix
df.corr()

# 7. Split data into train, test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=.2,random_state=10)

print('x_train shape: ',x_train.shape)
print('x_test shape: ',x_test.shape)
print('y_train shape: ',y_train.shape)
print('y_test shape: ',y_test.shape)

# 8. Find coefficients & intercept

slr = sm.OLS(y_train,x_train).fit()
slr.params

# 9.  Linear Relationship between feature and target
Plot the line with b1 and b0 as slope and y-intercept.

fig, ax = plt.subplots()

ax.axline((0,5.662523 ), slope=1.161558, color='C0', label='by slope')

# 10. Evaluation of model with scikit-learn


from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

#rmse on train and test data
print('rmse on train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))
print('rmse on test: ',np.sqrt(mean_squared_error(y_test,y_test_pred)))

#r2 for train and test

print('r2 on train: ', r2_score(y_train,y_train_pred))
print('r2 on test: ', r2_score(y_test,y_test_pred))

# 11. Calculate the accuracy of the model for both training and test data set

Hint: .score() function

y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

print('r2 on train: ', r2_score(y_train,y_train_pred))
print('r2 on test: ', r2_score(y_test,y_test_pred))
