# Import librarie
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# set seed
np.random.seed(123)

# simulate correlated data using the Cholesky matrix decomposition

num_samples = 10000

# Desired number of variables 
num_variables = 3

# The desired mean values of the sample.
mu = np.array([10, 9, 9])

# The desired correlation matrix.
r = np.array([[1, .8, .5],
               [.8, 1, .3],
               [.5, .3, 1]])

print("Desired correlation matrix:")
print(pd.DataFrame(r))

# We find the Cholesky decomposition of the covariance matrix, and multiply that by the matrix of uncorrelated random variables to create correlated variables..

linalg = np.linalg
L = linalg.cholesky(r)

uncorrelated = np.random.standard_normal((num_variables, num_samples))

correlated = np.dot(L, uncorrelated) + np.array(mu).reshape(3, 1)

# We create a dataframe with correlated data.

data = pd.DataFrame(correlated).T

data.columns = ['Y', 'X1', 'X2']

# We check the correlation matrix of the dataframe which is very similar to the desired one above.

print("Final correlation matrix from simulated data")
data.corr().round(3)

# function for calculating the sum of squares.

def sum_squares(x):
    ss = sum((x - np.mean(x))**2)
    return ss
    print(ss)


sum_squares_y = round(sum_squares(data.Y), 2)
sum_squares_x1 = round(sum_squares(data.X1), 2)

# Just looking at this plot, we can see that Y has less variation than X1. Neat.
venn2(subsets=(sum_squares_y,sum_squares_x1,0), set_labels = ('Y', 'X1'))

model = ols('Y ~ X1', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

# Now we can visualize the covariance between these two variables.

sum_squares_both_y_x1 = round(aov_table.sum_sq[0], 2)

y = sum_squares_y - sum_squares_both_y_x1
x1 = sum_squares_x1 - sum_squares_both_y_x1
y_x1 = sum_squares_both_y_x1

venn2(subsets=(y,x1,y_x1), set_labels = ('Y', 'X1', 'Y_X1'))

# We can however calculate R2 from the areas presented
r_squared = y_x1 / (y + y_x1)
r_squared

# We confirm the results of R2 as follows:
model.rsquared

#########################
# Multicolinearity Issue
#########################
plt.figure(figsize=(5, 5))
v = venn3(subsets=(4,4,2,4,2,2,2))

v.get_label_by_id('100').set_text('A')
v.get_label_by_id('010').set_text('B')
v.get_label_by_id('001').set_text('C')
v.get_label_by_id('110').set_text('D')
v.get_label_by_id('011').set_text('E')
v.get_label_by_id('101').set_text('F')
v.get_label_by_id('111').set_text('G')


#Calculation of areas: 

y_total = round(sum_squares(data.Y))    # A + D + F + G
x1_total = round(sum_squares(data.X1))  # B + D + E + G
x2_total = round(sum_squares(data.X2))  # C + E + F + G


# A

model = ols('Y ~ X2 + X1', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
y_alone = round(aov_table.sum_sq[2], 2)

# B

model = ols('X1 ~ Y + X2', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
x1_alone = round(aov_table.sum_sq[2], 2)

# C

model = ols('X2 ~ Y + X1', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
x2_alone = round(aov_table.sum_sq[2], 2)

# D + G

model = ols('Y ~ X1', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
y_plus_x1 = round(aov_table.sum_sq[0], 2)

# F + G

model = ols('Y ~ X2', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
y_plus_x2 = round(aov_table.sum_sq[0], 2)

# E + G

model = ols('X1 ~ X2', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
x1_plus_x2 = round(aov_table.sum_sq[0], 2)

# D = (A + D + F + G) − A − (F + G)

y_x1_alone = round(y_total - y_alone - y_plus_x2)

# E = (B + D + E + G) − B − (D + G)

y_x2_alone = round(y_total - y_alone - y_plus_x1)

# G = (D + G) − D

y_x1_x2_alone = round(y_plus_x1 - y_x1_alone)

# F = (F + G) - G

x1_x2_alone =  round(y_plus_x2 - y_x1_x2_alone, 2)

plt.figure(figsize=(5, 5))
v = venn3(subsets=(y_alone, x1_alone, y_x1_alone, x2_alone, y_x2_alone, x1_x2_alone, y_x1_x2_alone))

v.get_label_by_id('100').set_text('A')
v.get_patch_by_id('100').set_facecolor('White')
v.get_patch_by_id('100').set_edgecolor('Black')
v.get_patch_by_id('100').set_linestyle('-')
v.get_patch_by_id('100').set_linewidth(2)

v.get_label_by_id('010').set_text('B')
v.get_patch_by_id('010').set_facecolor('White')
v.get_patch_by_id('010').set_edgecolor('Black')
v.get_patch_by_id('010').set_linestyle('-')
v.get_patch_by_id('010').set_linewidth(2)

v.get_label_by_id('001').set_text('C')
v.get_patch_by_id('001').set_facecolor('White')
v.get_patch_by_id('001').set_edgecolor('Black')
v.get_patch_by_id('001').set_linestyle('-')
v.get_patch_by_id('001').set_linewidth(2)

v.get_label_by_id('110').set_text('D')
v.get_patch_by_id('110').set_facecolor('White')
v.get_patch_by_id('110').set_edgecolor('Black')
v.get_patch_by_id('110').set_linestyle('-')
v.get_patch_by_id('110').set_linewidth(2)

v.get_label_by_id('011').set_text('E')
v.get_patch_by_id('011').set_facecolor('White')
v.get_patch_by_id('011').set_edgecolor('Black')
v.get_patch_by_id('011').set_linestyle('-')
v.get_patch_by_id('011').set_linewidth(2)

v.get_label_by_id('101').set_text('F')
v.get_patch_by_id('101').set_facecolor('Red')
v.get_patch_by_id('101').set_edgecolor('Black')
v.get_patch_by_id('101').set_linestyle('-')
v.get_patch_by_id('101').set_hatch('/') 
v.get_patch_by_id('101').set_linewidth(2)

v.get_label_by_id('111').set_text('G')
v.get_patch_by_id('111').set_facecolor('Red')
v.get_patch_by_id('111').set_edgecolor('Black')
v.get_patch_by_id('111').set_linestyle('-')
v.get_patch_by_id('111').set_hatch('/') 
v.get_patch_by_id('111').set_linewidth(2)

# A
y_alone

# D
y_x1_alone

# G 
y_x1_x2_alone

# F
x1_x2_alone

# R2 = D + G + F / A + D + G + F
r_squared = (y_x1_alone + y_x1_x2_alone + x1_x2_alone)/(y_alone + y_x1_alone + y_x1_x2_alone + x1_x2_alone)

round(r_squared,2)


model = ols('Y ~ X1 + X2', data).fit()
round(model.rsquared,2)

########################
# Experiment on regression coefficients with multicollinearity
########################


numbers  = [float(x)/100 for x in range(92)]

parameters = []

for i in numbers:
    num_samples = 10000
    num_variables = 3
    mu = np.array([0, 0, 0])
    
    r = np.array([[1, .8, .5],
               [.8, 1, i],
               [.5, i, 1]])

    linalg = np.linalg
    L = linalg.cholesky(r)
    uncorrelated = np.random.standard_normal((num_variables, num_samples))
    correlated = np.dot(L, uncorrelated) + np.array(mu).reshape(3, 1)
    data = pd.DataFrame(correlated).T
    data.columns = ['Y', 'B1', 'B2']
    model = ols('Y ~ B1 + B2', data).fit()
    params = model.params
    parameters.append(params)
    
    
parameters = pd.DataFrame(parameters)
parameters['r'] = numbers
parameters

import matplotlib.pyplot as plt

plt.plot(parameters['r'], parameters['B1'], label = "Estimator 1")
plt.plot(parameters['r'], parameters['B2'], label = "Estimator 2")
plt.legend()