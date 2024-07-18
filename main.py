import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# -----SETTING UP OUR DATA-----

# Reading our dummy data csv file
df = pd.read_csv("train_data.csv")

# Encoding Employment status to numbers
# print(df['Employment Status'].value_counts())

emp_encode = {'Employment Status': {'Employed': 1, 'Unemployed': 2, 'Self-employed': 3, 'Retired': 4}}
df.replace(emp_encode, inplace=True)
# print(df['Employment Status'].value_counts())

# Encoding education level
df['Education Level'].fillna('None', inplace=True)
# print(df['Education Level'].value_counts())

edulevel_encode = {'Education Level': {'Secondary': 2, 'Primary': 1, 'Higher': 3, 'Graduate': 4, 'None': 5}}
df.replace(edulevel_encode, inplace=True)
# print(df['Education Level'].value_counts())

# Encoding Region
# print(df['Region'].value_counts())

region_encode = {'Region': {'Urban': 1, 'Rural': 2, 'Suburban': 3}}
df.replace(region_encode, inplace=True)
# print(df['Region'].value_counts())

# Encoding Owns Property
# print(df['Owns Property'].value_counts())

own_encode = {'Owns Property': {'Yes': 1, 'No': 0}}
df.replace(own_encode, inplace=True)
# print(df['Owns Property'].value_counts())

# Encoding Loan status
# print(df['Loan Status'].value_counts())

loan_encode = {'Loan Status': {'No Loan': 0, 'Pending': 1, 'Defaulted': 2, 'Paid': 3}}
df.replace(loan_encode, inplace=True)
# print(df['Loan Status'].value_counts())

# Encoding Eligibilty
# print(df['Eligibility'].value_counts())

eli_encode = {'Eligibility': {'Yes': 1, 'No': 0}}
df.replace(eli_encode, inplace=True)
# print(df['Eligibility'].value_counts())

# TRAINING

KNN = KNeighborsClassifier()

x = df[['Income', 
        'Number of Dependents', 
        'Employment Status', 
        'Age', 
        'Education Level', 
        'Region', 
        'Owns Property', 
        'Bank Balance', 
        'Loan Status', 
        'Medical Expenses']]

y = df['Eligibility']

# print(x.head())
# print(y.head())

KNN = KNN.fit(x,y)

test = pd.DataFrame()

# TESTING

# Reading our dummy test file

test_df = pd.read_csv('test_data.csv')

test['Income'] = test_df['Income']
test['Number of Dependents'] = test_df['Number of Dependents']
test['Employment Status'] = test_df['Employment Status']
test['Age'] = test_df['Age']
test['Education Level'] = test_df['Education Level']
test['Region'] = test_df['Region']
test['Owns Property'] = test_df['Owns Property']
test['Bank Balance'] = test_df['Bank Balance']
test['Loan Status'] = test_df['Loan Status']
test['Medical Expenses'] = test_df['Medical Expenses']

# print(x)

predict_eligibilty = KNN.predict(x)

for prediction in predict_eligibilty:
    if prediction == 1:
        print('Yes')
    else:
        print('No')

# ------ Model Done -------

# ------- Representation of our model ----------

yes_count = np.sum(predict_eligibilty == 1)
no_count = np.sum(predict_eligibilty == 0)

# Prepare the data for the bar plot
labels = ['Yes', 'No']
counts = [yes_count, no_count]

# Create the bar plot
plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Eligibility')
plt.ylabel('Number of Predictions')
plt.title('Number of Yes and No Predictions')
plt.show()
