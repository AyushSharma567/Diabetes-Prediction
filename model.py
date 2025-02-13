#........................importing header files ...........................
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import pickle

# ........................Importing Dataset ...............................
data = pd.read_csv('./diabetes_prediction_dataset.csv')


#.........................Removing duplicate data..........................
data.drop_duplicates(inplace=True)

label_encoder = preprocessing.LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

# Convert smoking history to numerical format
smoking_history_mapping = {'never': 0, 'No Info': -1, 'current': 2, 'former': 1, 'ever': 2, 'not current': 0}
data['smoking_history'] = data['smoking_history'].map(smoking_history_mapping)

data = data[data['age'].mod(1) == 0]
data['age'] = data['age'].astype(int)

#.........................Spliting Dataset into training and testing and training model .......
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

(X_sample, y_sample) = train_test_split(data, train_size=0.8, stratify = y)

X_train = X_sample.iloc[:,:-1].values
y_train = X_sample.iloc[:,-1].values
X_test_smote = y_sample.iloc[:,:-1].values
y_test_smote = y_sample.iloc[:,-1].values


smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rfc_smote = RandomForestClassifier()
rfc_smote.fit(X_train_smote, y_train_smote)
y_pred_rfc_smote = rfc_smote.predict(X_test_smote)

pickle.dump(rfc_smote,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

df=pd.read_pickle("model.pkl")
print(df)

# person_X = stand.transform([[1,23.0, 0,0,0,22.9,5.4, 108]])
# person_predict = rfc.predict(person_X)
# person_predict = (person_predict>0.5)
# print(person_predict)