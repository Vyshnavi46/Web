# <center>Website Behavior Analysis</center>

## Index
### 1. Import Required Libraries
### 2. Analysis the WebSite Browing Behavior csv file
### 3. Analysis the final conversions csv file
### 4. Feature Engineering (Variable Imputation)
### 5. Model Selection Criteria (Basis of choosing the final Technique)
### 6. Measurement Criteria (Comparison of Various Models)
### 7. Scope for improvement
### 8. Save model to File using pickle

## 1. Import Required Libraries

# Import library for Read file, Analysis data, viz 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ignore the warning for future package and deprecated package
import warnings
warnings.filterwarnings('ignore')

# import library for Model selection model from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import the library for classifier model from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Import the library for test the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Import the library for roc curve
from sklearn.metrics import roc_auc_score, roc_curve

# import the pickle model for save the model
import pickle

## 2. Analysis the WebSite Browing Behavior csv file

# Read the site_browingbehavior file using pandas
wb = pd.read_csv('Problem2_Site_BrowingBehavior.csv', sep = '\t', header = None)
wb.columns = ['Timestamp', 'UserID', 'Website_section_visited']

wb.head()

# Checking the wb dataframe info and non values
print(wb.info())
print(wb.isnull().sum())

### 2.1 Analysis the Site_BrowingBehavior.csv file

# copy the dummy dataframe for analysis 
wb_dummy = wb.copy()

# Add a new column for diff the guest user and registered user
wb_dummy['isGuest'] = wb_dummy['UserID'].apply(lambda g: g <= 0 )

wb_dummy.head()

# Add the new column for analysis the browsing behaviour dataframe
wb_dummy['TimeOfDay'] = wb_dummy['Timestamp'].apply(lambda x:x[11:16])
wb_dummy.sample()

# User visited time with time 
wb_analysis= wb_dummy.groupby('TimeOfDay')['Timestamp'].count().reset_index().rename(columns={'Timestamp': 'UserCount'})
wb_analysis= wb_analysis.sort_values("TimeOfDay")
wb_analysis.plot(x='TimeOfDay',y='UserCount',figsize=(15,5))

### 2.2 Guest User Analysis

# Guest user analysis
print('Total rows & columns : ',wb_dummy.shape)
print('Total users : ', len(wb_dummy.UserID.unique()))

wbuser= pd.DataFrame()
wbuser['count'] = wb_dummy.groupby('isGuest').size()
wbuser['percentile']= wbuser['count'].apply(lambda x:x/len(wb_dummy)*100)
wbuser

# Guest user plot
wbuser.plot.bar(x='count', y='percentile',figsize=(5,5),color = 'r');
plt.ylabel('Percentage')
plt.xlabel('User Count')
plt.title('Product purchased by User and GuestUser')
plt.show();

### 2.3 Website Visited Analysis

# Registred users are on which product
wb_websiteAnalysis= wb_dummy[wb_dummy['isGuest']==False].groupby('Website_section_visited')['Timestamp'].count().reset_index().rename(columns={'Timestamp': 'UserCount'})
wb_websiteAnalysis['Percentile']= wb_websiteAnalysis['UserCount'].apply(lambda x:x/len(wb_dummy)*100)
wb_websiteAnalysis = wb_websiteAnalysis.sort_values("UserCount",ascending=False)
wb_websiteAnalysis

wb_websiteAnalysis.plot.bar(x='Website_section_visited', y='UserCount', rot=70,figsize=(15,5));
plt.title('Website visited grouped by User')
plt.xlabel('Website Visited Categories')
plt.ylabel('User Count')
plt.show;

# All the website catagories with visited time
for i in wb_dummy.Website_section_visited.unique():
    wb_TimeAnalysis = wb_dummy[wb_dummy['Website_section_visited']==i].groupby('TimeOfDay')['Timestamp'].count().reset_index().rename(columns={'Timestamp': 'UserCount'})
    wb_TimeAnalysis.plot(x='TimeOfDay',y='UserCount',figsize=(15,5),title = i )

### 2.4 Count the Website visited by User 

# Count the website visited by user
wb_dummy.groupby('UserID')['Website_section_visited'].value_counts().sort_values(ascending = False).head(20)

## 3. Analysis the final conversions csv file

# Read the final conversions file using pandas
fc = pd.read_csv('Problem2_FInalConversions.csv', sep = '\t', header = None)
fc.columns = ['Timestamp', 'UserID','ProductCode', 'CartValue']

fc.info()

fc.isnull().sum()

### 3.1 Analysis the final conversions file

# copy the dummy dataframe for analysis 
fc_dummy = fc.copy()

# Add a new column for guest user purchased
fc_dummy['isGuest'] = fc_dummy['UserID'].apply(lambda f: f <= 0 )

fc.head()

# Guest user and normal user
print('Total rows & columns : ',fc_dummy.shape)
print('Total users : ', len(fc_dummy.UserID.unique()))

fcuser = pd.DataFrame()
fcuser['count'] = fc_dummy.groupby('isGuest').size()
fcuser['percentile']= fcuser['count'].apply(lambda x:x/len(fc_dummy)*100)
fcuser


fcuser.plot.bar(x='count', y='percentile',figsize=(5,5),color= 'r');
plt.ylabel('Percentage')
plt.xlabel('User Count')
plt.title('Product purchased by User and GuestUser')
plt.show();

# Added the new column for time
fc_dummy['TimeOfDay'] = fc_dummy['Timestamp'].apply(lambda x:str(x)[11:16])
fc_dummy.sample(5)

### 3.2 User Count for each product by Guest User and Registered User

#### 3.2.1 Guest User for each product

# Guest User Analysis with each product
fc_GuestAnalysis = fc_dummy[fc_dummy['isGuest']==True].groupby('ProductCode')['Timestamp'].count().reset_index().rename(columns={'Timestamp': 'UserCount'})
fc_GuestAnalysis['Percentile']= fc_GuestAnalysis['UserCount'].apply(lambda x:x/len(fc_dummy)*100)
fc_GuestAnalysis = fc_GuestAnalysis.sort_values('UserCount',ascending=False)
fc_GuestAnalysis.head()

#### 3.2.2. Registered User for each product

# Registered User analysis with product
fc_NonGuestAnalysis = fc_dummy[fc_dummy['isGuest']==False].groupby('ProductCode')['Timestamp'].count().reset_index().rename(columns={'Timestamp': 'UserCount'})
fc_NonGuestAnalysis['Percentile']= fc_NonGuestAnalysis['UserCount'].apply(lambda x:x/len(fc_dummy)*100)
fc_NonGuestAnalysis = fc_NonGuestAnalysis.sort_values('UserCount',ascending=False)
fc_NonGuestAnalysis.head()

### 3.3 Top Selling Product 

#### 3.3.1 Top selling product by Registered user

# Top Selling product
fc_NonGuestTopProduct =fc_NonGuestAnalysis.iloc[:10]
fc_NonGuestTopProduct

#### 3.3.2 Top selling product by Guest user

# Top Selling Product
fc_GuestTopProduct =fc_GuestAnalysis.iloc[:5]
fc_GuestTopProduct

## 4. Feature Engineering (Variable Imputation)

# Create Dummies for website section visited column
dummies = pd.get_dummies(wb.Website_section_visited)
dummies

### 4.1 Dummies added to wb dataframe

# Drop the column if exists
if 'Timestamp' in wb.columns:
    wb.drop('Timestamp',axis = 1, inplace=True)
if 'Website_section_visited' in wb.columns:
    wb.drop('Website_section_visited',axis = 1, inplace=True)

# concat the two dataframe
wb_tmp = pd.concat([wb, dummies], axis= 'columns')

# groupby and sum the column using UserID column
wb_tmp = wb_tmp.groupby('UserID').sum().reset_index()

wb_tmp

### 4.2 Dummies added to fc dataframe

# Drop the column if exists
if 'Timestamp' in fc.columns:
    fc.drop('Timestamp',axis = 1, inplace=True)
if 'ProductCode' in wb.columns:
    wb.drop('ProductCode',axis = 1, inplace=True)

# groupby and sum the column using UserID column
fc_tmp = fc.groupby('UserID').sum().reset_index()
fc_tmp

### 4.3 Merging the wb and fc dataframe

# Merge the both dataframe using right merge operation
df = pd.merge(fc_tmp, wb_tmp ,on='UserID',how='right')
df = df.fillna(0)
df['isPurchased'] = df['CartValue'].apply(lambda x : 1 if x > 0 else 0)
df

# describe the final dataframe
df.describe()

df[(df['isPurchased']==1)]
for c in df.columns:
    if c not in ['UserID','isPurchased']:
        df.loc[0,c]=df.loc[0,c]/1407879
df

# Analysis the isPurchased column
df_purchased = pd.DataFrame()
df_purchased['count'] = df.groupby('isPurchased').size()
df_purchased['percentile']= df_purchased['count'].apply(lambda x:x/len(df)*100)
df_purchased

# Analysis the isPurchased column using histogram
df_purchased.plot.bar(x='count', y='percentile',figsize=(5,5));
plt.ylabel('Percentage')
plt.xlabel('User Count')
plt.title('Product purchased by User and GuestUser')
plt.show();

# correletion method
df.corr()

# Heatmap
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(dummies.corr(), annot=True);

#### Most user visited pages only picked for analysis

# X = df.drop('isPurchased',axis = 'columns')
X = df.loc[:,[
    'product',
    'product-listing-category',
    'home',
    'default',
    'content',
    'iroa',
    'cart',
    'product-listing-search'
]]
y = df.isPurchased

X

## 5. Model Selection Criteria (Basis of choosing the final Technique)

### 5.1 Train Test the model

# Train Test Split method
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.25, random_state= 0)

print(X.shape,X_train.shape,X_test.shape)

### 5.2 Apply the Standard Scaler method

# Apply Standard Scaler method
sd = StandardScaler()
X_train = sd.fit_transform(X_train)
X_test = sd.transform(X_test)

## 6. Measurement Criteria (Comparison of Various Models)

### 6.1 Logistics Regression

# Create a new object for logistic Regression
lm = LogisticRegression()

# fit model
lm.fit(X_train, y_train)

# print the accuracy score logistic regression
print('Accuracy score of the training data : ', lm.score(X_test,y_test))

# Confusion Matrix of Logistic Regression
cm_lr = confusion_matrix(y_test,lm.predict(X_test))

f, ax = plt.subplots(figsize = (6,5))
sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax, cmap='magma')
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Logistic Regression")
plt.show()

### 6.2 Support Vector Machine Classifier

#svc = svm.SVC(kernel='linear')

#svc.fit(X_train,y_train)

#X_train_prediction = svc.predict(X_train)
#training_data_accuray = accuracy_score(X_train_prediction,y_train)

#print('Accuracy on training data : ', training_data_accuray)

#cm_lr = confusion_matrix(y_test,svc.predict(X_test))

#f, ax = plt.subplots(figsize = (6,5))
#sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax, cmap='magma')
#plt.xlabel("y_predicted")
#plt.ylabel("y_true")
#plt.title("Confusion Matrix of SVM Regression")
#plt.show()

### 6.3 Decision Tree Classifier

# Create a object for DecisionTreeClassifier and fit the model
dt = DecisionTreeClassifier(max_depth = 4, random_state = 0)
dt.fit(X_train, y_train)

# Score the decision tree model
dt.score(X_test, y_test)

# Print the model
print('Accuracy score of the training data : ', dt.score(X_test,y_test))

# Confusion Matrix
cm_lr = confusion_matrix(y_test,dt.predict(X_test))

f, ax = plt.subplots(figsize = (6,5))
sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax, cmap='magma')
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of SVM Regression")
plt.show()

## 6.4 Checking Probability
### lm.predict_proba([[ 'product', 'product-listing-category', 'home', 'default', 'content', 'iroa', 'cart', 'product-listing-search' ]])

# Checking the probabilty using selected columns in logistic regression
lm.predict_proba([[30,34,9,9,9,3,3,0]])

### dt.predict_proba([[ 'product', 'product-listing-category', 'home', 'default', 'content', 'iroa', 'cart', 'product-listing-search' ]])

# Checking the probabilty using selected columns in decisionTree
dt.predict_proba([[30,34,9,9,9,3,3,0]])

## 7. Scope for improvement

# Roc Curve
from sklearn.metrics import roc_auc_score, roc_curve

Log_ROC_aur = roc_auc_score(y,lm.predict(X))
fpr,tpr, thresholds = roc_curve(y, lm.predict_proba(X) [:,1])

plt.figure()
plt.plot(fpr,tpr, label = 'Logit Model 1 (area = %0.2f' % Log_ROC_aur)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

X2 = df.loc[:,[
    'CartValue',
    'product',
    'product-listing-category',
    'home',
]]
y2 = df.isPurchased

X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2, test_size= 0.25, random_state= 0)

print(X2.shape,X2_train.shape,X2_test.shape)

lm2 = LogisticRegression()
lm2.fit(X2_train, y2_train)

print('Accuracy score of the training data : ', lm2.score(X2_test,y2_test))

Log_ROC_aur2 = roc_auc_score(y2,lm2.predict(X2))
fpr2,tpr2, thresholds2 = roc_curve(y2, lm2.predict_proba(X2) [:,1])

plt.figure()
plt.plot(fpr,tpr, label = 'Logit Model 1 (area = %0.2f' % Log_ROC_aur)
plt.plot(fpr,tpr, label = 'Logit Model 2 (area = %0.2f' % Log_ROC_aur2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

## 8. Save model to File using pickle

### 8.1 Save model

# Save the logistic model using pickle
with open('logistic_model', 'wb') as f:
    pickle.dump(lm,f)

# Save the decision model using pickle
with open('DecisionTree_model', 'wb') as f:
    pickle.dump(dt,f)

### 8.2 Load model

# Load the logistic model using pickel
with open('logistic_model', 'rb') as f:
    logistic_model = pickle.load(f)

# Load the decisiontree model using pickel
with open('DecisionTree_model', 'rb') as f:
    DecisionTree_model = pickle.load(f)

### 8.3 Checking model probability

#### 8.3.1 Logistic Model

# Checking Probability Logistic model
lm.predict_proba([[9,1,1,1,1,1,1,1]])

logistic_model.predict_proba([[9,1,1,1,1,1,1,1]])

#### 8.3.1 Decision Model

# Checking Probability decision tree model
dt.predict_proba([[9,1,1,1,1,1,1,1]])

DecisionTree_model.predict_proba([[9,1,1,1,1,1,1,1]])
