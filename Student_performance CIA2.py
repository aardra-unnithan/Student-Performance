#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm



# In[14]:


std_prfmnce =pd.read_csv("C:/Users/ardra/Downloads/archive (14)/StudentPerformanceFactors.csv")
std_prfmnce


# In[15]:


print(std_prfmnce.columns)


# In[16]:


print("\nDescriptive Statistics:")
print(std_prfmnce.describe())


# In[17]:


plt.figure(figsize=(8, 6))
sns.regplot(x=std_prfmnce['Attendance'], y=std_prfmnce['Exam_Score'], scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Exam Score vs Attendance Percentage")
plt.show()

print("The correlation between the two variables is")
print(std_prfmnce['Attendance'].corr(std_prfmnce['Exam_Score']))


# In[20]:


# Prepare Data for Regression
X = std_prfmnce[['Attendance']]  # Independent variable
Y = std_prfmnce['Exam_Score']  # Dependent variable

X = sm.add_constant(X)
X.head()
model = sm.OLS(Y,X, missing='drop')
model_result=model.fit()
model_result.summary()


# In[22]:


fitted_values = model_result.fittedvalues
residuals = model_result.resid

plt.figure(figsize=(8,5))
plt.scatter(fitted_values, residuals, alpha=0.8)
plt.axhline(y=0, color='r', linestyle='dashed')
plt.xlabel('fitted_values')
plt.ylabel('Residuals')
plt.title('Residuals Vs. Fitted Values')
plt.show()


# In[24]:


import statsmodels.stats.diagnostic as smd
import numpy as np

# Get residuals and independent variables (exog)
residuals = model_result.resid
exog = model_result.model.exog  # Independent variables (including constant)

# Perform the Breusch-Pagan test
bp_test = smd.het_breuschpagan(residuals, exog)
p_value = bp_test[1]  # Extract the p-value

print(f"Breusch-Pagan test p-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Heteroscedasticity detected! Consider transformation or robust standard errors.")
else:
    print("No significant heteroscedasticity detected.")


# In[ ]:




