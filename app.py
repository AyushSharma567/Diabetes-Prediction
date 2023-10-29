from flask import Flask
from flask import request
from flask import url_for
from flask import redirect
from flask import render_template
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')
            
        #  0   gender               94133 non-null  int64  
        #  1   age                  94133 non-null  int64  
        #  2   hypertension         94133 non-null  int64  
        #  3   heart_disease        94133 non-null  int64  
        #  4   smoking_history      94133 non-null  int64  
        #  5   bmi                  94133 non-null  float64
        #  6   HbA1c_level          94133 non-null  float64
        #  7   blood_glucose_level  94133 non-null  int64  
        #  8   diabetes             94133 non-null  int64  

@app.route('/predict',methods=['POST'])
def predict():
        if request.method == 'POST':
            Dependencies = int(request.form['gender'])
            Education =int(request.form['age'])
            Selfemp =int(request.form['hyp'])
            incomeanumn=int(request.form['hd'])
            loananumn=int(request.form['sh'])
            loanterm=float(request.form['bmi'])
            cibilscore=float(request.form['hl'])
            masset=int(request.form['bgl'])
            final_features = np.array([Dependencies,Education,Selfemp,incomeanumn,loananumn,loanterm,cibilscore,masset])
            print(final_features)

            final_features = final_features.reshape(1,-1)
            person_predict = model.predict(final_features)
            print(person_predict )
           
            if person_predict.item() >= 0.7 :
                return render_template('index1.html', prediction_text='you have high chance of getting Diabetes. ')
            else:
                return render_template('index1.html',prediction_text='you have low chance of getting Diabetes.  ')


if __name__ == "__main__":
    app.run(debug=True)