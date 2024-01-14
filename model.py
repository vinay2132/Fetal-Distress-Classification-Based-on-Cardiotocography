import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import pickle

df = pd.read_csv("fetal_health.csv")
X=df.drop(["fetal_health"],axis=1)
y=df["fetal_health"]
col_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df= scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)

X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
pred_rf = rf.predict(X_test)
accuracy = accuracy_score(y_test, pred_rf)
print(accuracy)


pickle.dump(rf, open('modelrf.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


model = pickle.load(open('modelrf.pkl', 'rb'))
print(model)



