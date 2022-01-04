#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import tree

#open file

HF=pd.read_csv('heart_failure_clinical_records_dataset.csv')

#First five rows

print(HF.head(5))
print("the length of dataset is ",len(HF))
print("nbre of Columns in this dataset", HF.shape[1])

#missing value

print ("the nbre of missing value in each col",HF.isnull().sum())

#describe data

print(HF.describe())
HF.boxplot()
plt.show()
print(HF.columns)

#rename the column

HF=HF.rename(columns={'creatinine_phosphokinase':'CPK','ejection_fraction':'EF','high_blood_pressure':'HBP'})
print(HF.columns)


## Step two Data Visualization

#Data Visualization

plt.figure(figsize=(8, 8))
plt.pie(HF['DEATH_EVENT'].value_counts(), labels=["Heart Disease", "No Heart Disease"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()
plt.pie(HF['smoking'].value_counts(), labels=["smoking", "no smoking"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()
plt.pie(HF['sex'].value_counts(), labels=["femal", "mal"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()
plt.pie(HF['diabetes'].value_counts(), labels=["diabetes", "no diabetes"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()
plt.pie(HF['HBP'].value_counts(), labels=["high_blood_pressure", "no high_blood_pressure"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()
plt.pie(HF['anaemia'].value_counts(), labels=["anaemia", "no anaemia"], autopct='%.1f%%', colors=['#36a2ac', '#413f80'])
plt.show()

#Histogramme

HF.hist()
plt.show()

###Step three :Data analysis
from pandas.plotting import scatter_matrix
sns.scatterplot(data=HF, x='serum_creatinine', y='serum_sodium', hue='DEATH_EVENT')
plt.title('serum_creatinine vs serum_sodium')
plt.show()
#Data analysis
sns.lineplot(data=HF, x='serum_creatinine',y='platelets', color='blue')
anaemia_HF2 = HF.groupby('age')[['anaemia']].count().sort_values('anaemia', ascending=False)
print("c'est le grp de anaemia",anaemia_HF2) ######
plt.figure(figsize=(8,8))
plt.title("Anaemia of  different ages")
plt.xlabel('Age',fontsize=10);
plt.ylabel('Anaemia Count',fontsize=10)
plt.scatter(anaemia_HF2.index,anaemia_HF2,c="red")

#correlation

hfcorr=HF.corr()
plt.figure(figsize=(10,10))
sns.heatmap(hfcorr)

#Separate the data

#numerical_features = ["age", "CPK", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
#categorical_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

##relationship of platelets count and death

def ef_category(ejection_fraction):
  if ejection_fraction == 50 and ejection_fraction <= 70:
    return 'Normal ejection_fraction'
  elif ejection_fraction <=40:
    return 'Low  ejection_fraction'
  elif ejection_fraction == 41 and ejection_fraction <=49:
    return 'Borderline  ejection_fraction'
  elif ejection_fraction > 75:
    return 'High  ejection_fraction'
  else:
    return 'NA'
HF['EF_category'] = HF['EF'].apply(lambda EF: ef_category(EF))
HF["gender"]=HF["sex"].apply(lambda toLabel: 'F' if toLabel ==0 else 'M')
plt.figure(figsize=(15,7))
plt.title("Gender of Smoking",fontweight=200,fontsize=20)
gender_smoking = HF['gender'].loc[HF['smoking']==1].value_counts()
plt.pie(gender_smoking, labels=gender_smoking.index, autopct='%1.1f%%', startangle=180,colors =['green','yellow' ]);
EF_avg_age = HF.groupby(['EF_category'])[['age']].mean()
EF_avg_age.plot(figsize=(15,7),kind='barh',color = 'pink').set_title("Average age of respondents in EF categories")




###################################################################################################################################


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.info()
print(df.describe())
print(df.head(5))
print(df.columns)

df=df.rename(columns={'creatinine_phosphokinase':'CPK','ejection_fraction':'EF','high_blood_pressure':'HBP'})
print(df.head(5))
print("the length of dataset is ",len(df))

#missing value
print(" the nbre of missing value in each col", df.isnull().sum())

#Correlation
corrmat  = df.corr()
print(corrmat)
top_corr_features = corrmat.index
plt.figure(figsize=(20,10))
g = sns.heatmap(df[top_corr_features].corr(),annot = True,cmap = "RdYlGn")

#As we can see that none of the features are highly correlated (not greater than 0.5) we cannot remove any columns.

#Spliting_The_Data:
###################
features = ['age', 'anaemia', 'CPK', 'diabetes',
       'EF', 'HBP', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
label = ['DEATH_EVENT']
X = df[features]
y = df[label]
print("This is the part heat",y)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.33, shuffle =True,random_state=42)
print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of Y_train:', y_train.shape)
print('Shape of Y_test:', y_test.shape)

features1 = ['age', 'CPK',
       'EF',  'platelets',
        'serum_sodium', 'time']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(X_train[features1]) ###??????????????
X_test = sc.transform(X_test[features1])

#classification using LogisticRegression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix, auc, classification_report,roc_curve
classification_report = classification_report(y_test, y_pred_log_reg)
print(classification_report)
print(confusion_matrix(y_test, y_pred_log_reg))
auc = roc_auc_score(y_test, y_pred_log_reg)
auc
#fonction : calcule du matrice de confusion
cm = confusion_matrix(y_test, y_pred_log_reg)
tn = cm[0,0]
fp = cm[0,1]
tp = cm[1,1]
fn = cm[1,0]
accuracy  = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1score  = 2 * precision * recall / (precision + recall)
print("Accuracy de Log_Reg est :",accuracy)
print("c'est le score dans Log_Reg:",f1score)
print("c'est la precision dans Log_Reg:",precision)
print("c'est la Recall dans Log_Reg:",recall)
#classification using DecisionTreeClassifier
dt_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)#3 niveaux #question to Kheir el din ?
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print("C'est le matrice :",confusion_matrix(y_test, dt_pred))
cm = confusion_matrix(y_test, dt_pred)
tn = cm[0,0]
fp = cm[0,1]
tp = cm[1,1]
fn = cm[1,0]
accuracy  = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1score  = 2 * precision * recall / (precision + recall)
print("Accuracy de DTree est :",accuracy)
print("c'est le score:",f1score)
print("c'est la precision:",precision)
print("c'est la Recall:",recall)
dt_f1 = f1_score(y_test, dt_pred)
dt_f1
#fini shuf mn khilel lmukarane ben accuracy wl precision bl logisticRegression w ben DescicionTree mn le2i ennu LogisticRegression heyi lbest classifier

#hiearchie
dt_clf=dt_clf.fit(X_train,y_train)
dt_clf.score(X_train,y_train)
print(dt_clf.score(X_train,y_train))
from six import StringIO

with open("heart_failure_clinical_records_dataset.dot","w") as f:
    f=tree.export_graphviz(dt_clf,out_file=f,feature_names=features)


