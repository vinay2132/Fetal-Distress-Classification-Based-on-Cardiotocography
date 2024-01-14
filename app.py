from flask import Flask,render_template,request
import numpy as np
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

model = pickle.load(open('modelrf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():

    features = [float(x) for x in request.form.values()]
    data = features
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    labels = ['Baseline Value', 'Accelerations', 'Fetal Movement',
       'Uterine Contractions', 'Light Decelerations', 'Severe Decelerations',
       'Prolongued Decelerations', 'ASTV',
       'MSTV',
       'ALTV',
       'MLTV', 'Width',
       'Min', 'Max', 'No: of peaks',
       'No: of zeros', 'Mode', 'Mean',
       'Median', 'Variance', 'Tendency',
       'Fetal Status']
    data1 = data[4:7]
    labels1 = labels[4:7]
    data2 = data[7:11]
    labels2 = labels[7:11]
    data31 = data[11:14]
    data32 = data[16:20]
    data3 = data31+data32
    labels31 = labels[11:14]
    labels32 = labels[16:20]    
    labels3 = labels31 + labels32
    data5 = data[1:4]
    labels5 = labels[1:4]
    data41 = [100,110,160]
    data41.append(data[0])
    data4 = data41
    labels41 = ['Pathological','Suspecious','Normal','Fetal Heartrate']
    labels4 = labels41
    
    return render_template("test.html",values1=data1,labels1=labels1,values2=data2,labels2=labels2,values3=data3,labels3=labels3,values4=data4,labels4=labels4,values5=data5,labels5=labels5,prediction=prediction[0])
    
if __name__ == '__main__':
    app.run()
