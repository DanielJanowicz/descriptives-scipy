## Importing packages
import pandas as pd
import tableone as t1
import researchpy as rp
import numpy as np

data = pd.read_csv('data\data1.csv')

## Creating an array
dfa = np.linspace(-6, 6, 20)
sin_dfa = np.sin(dfa)
cos_dfa = np.cos(dfa)

pd.DataFrame({'dfa': dfa, 'sin': sin_dfa, 'cos': cos_dfa}) 

## Manipulating data
data.shape
data.columns
print(data['Group'])
data[data['Group'] == 'Active']['Age'].mean()

## Groupby
groupby_group = data.groupby("Group")
for group, value in groupby_group['Age']:
    print((group, value.mean()))

groupby_group.mean()

## Plotting Data
from pandas.plotting import plotting # Refer to Errors folder
plotting.scatter_matrix(data[['Group', 'Age', 'Smoke']])

## Hypothesis Testing
from scipy import stats
stats.ttest_1samp(data['Age'], 0)

# T-testing for difference
group1_viq = data[data['Group'] == 'Active']['Age']
group2_viq = data[data['Group'] == 'Control']['Age']
stats.ttest_ind(group1_viq, group2_viq)

## Linear Regression
x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3 * x + 4 * np.random.normal(size=x.shape)
data = pd.DataFrame({'x': x, 'y': y})

# OLS model
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()

print(model.summary())

## Categorical Variables
data = pd.read_csv('data\data1.csv')
model = ols("Age ~ Group + 1", data).fit()
print(model.summary())

model = ols("Age ~ C(Group)", data).fit()

## T-test and HR & SBP
data_hr = pd.DataFrame()({'iq': data['HR'], 'type': 'hr'})
data_sbp = pd.DataFrame()({'iq': data['SBP'], 'type': 'sbp'})
data_long = pd.concat((data_hr, data_sbp))
print(data_long)

model = ols("iq ~ type", data_long).fit()
print(model.summary())

## Multiple Regression
data = pd.read_csv('data\iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary()) 

# ANOVA Testing
print(model.f_test([0, 1, -1, 0]))

## Seaborn for Statistical Data Visualization
import seaborn 
from tableone import TableOne, load_dataset
data = load_dataset("pn2012")
seaborn.pairplot(data, vars=["Age", "LOS"], kind='reg')

seaborn.pairplot(data, vars=["Age", "LOS"], kind='reg', hue='LOS')

## Matplotlib Settings
from matplotlib import pyplot as plt
plt.rodefaults()

## Implot for Regression
seaborn.lmplot(y='Age', x='LOS', data=data)

## Testing for Interaction
result = data.ols(formula='age ~ los + los', data=data).fit()
print(result.summary())

