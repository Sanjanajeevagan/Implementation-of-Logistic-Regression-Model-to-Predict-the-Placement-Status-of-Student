# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing – Load dataset, remove unnecessary columns, handle null/duplicate values, and encode categorical variables using LabelEncoder.
2. Feature & Target Selection – Split dataset into input features (x) and target label (y).
3. Train-Test Split – Divide the data into training and testing sets using train_test_split.
4. Model Training & Prediction – Train a Logistic Regression model with training data and predict on test data.
5. Evaluation – Evaluate performance using accuracy, confusion matrix, and classification report; then use the model to predict new inputs.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sanjana J
RegisterNumber:  212224230240


import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
TOP 5 ELEMENTS
<img width="1319" height="435" alt="Screenshot 2025-09-21 133458" src="https://github.com/user-attachments/assets/1112fedb-4c70-41c4-86d8-a927137225da" />

<img width="1215" height="358" alt="Screenshot 2025-09-21 133506" src="https://github.com/user-attachments/assets/2e99f0e3-9808-4402-b126-53df285788ce" />

<img width="412" height="396" alt="Screenshot 2025-09-21 133511" src="https://github.com/user-attachments/assets/800f9a2d-ed9e-48f7-b311-b46a90b3d356" />

Data Duplicate

<img width="285" height="113" alt="Screenshot 2025-09-21 133546" src="https://github.com/user-attachments/assets/73e98efb-7e39-42cb-96ea-dbbb0dd75532" />

Print Data

<img width="1028" height="775" alt="Screenshot 2025-09-21 133557" src="https://github.com/user-attachments/assets/58fb686e-639d-40a5-8774-30bc5e6f1878" />

Data Status

<img width="993" height="625" alt="Screenshot 2025-09-21 133604" src="https://github.com/user-attachments/assets/6057c5fc-d0f5-4dcf-8a1d-f4e542280502" />

Y prediction

<img width="579" height="370" alt="Screenshot 2025-09-21 133610" src="https://github.com/user-attachments/assets/8b4f0f7e-fafe-4dd9-b0c6-7f7a0522b5ef" />

confusion array

<img width="822" height="282" alt="Screenshot 2025-09-21 133617" src="https://github.com/user-attachments/assets/3e0ce36e-affa-4606-8cfe-019eee6df38c" />

Accuracy value

<img width="455" height="133" alt="Screenshot 2025-09-21 133622" src="https://github.com/user-attachments/assets/422d639a-1493-46d0-bf8f-52db4a3e5b37" />

Classification report

<img width="604" height="292" alt="Screenshot 2025-09-21 133631" src="https://github.com/user-attachments/assets/fa0efb2d-e58c-45fe-a3d0-913c9c33a024" />

Prediction of LR

<img width="286" height="43" alt="Screenshot 2025-09-21 133658" src="https://github.com/user-attachments/assets/2d6556c6-99e7-437c-98ee-32a646036cef" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
