from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
from logging import FileHandler,WARNING
from function import ajouter_donnees_au_csv ,calculate_bmi
from kmodes.kmodes import KModes
from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
from logging import FileHandler,WARNING
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from function import ajouter_donnees_au_csv
import urllib.request
import json
import os
import ssl


application = Flask(__name__,template_folder='templates')
app=application
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

#scaler=pickle.load(open('/config/workspace/Notebook/StandardScaler.pkl', 'rb'))
#model = pickle.load(open('c:/Users/talla/Music/Memoire/memoireApp/memoireknn_model.pkl', 'rb'))
file_path1   = 'code'
model = load_model(file_path1)
## Route for homepage

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/index')
def home():
    return render_template('index.html')




## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    output=""

    if request.method=='POST':

        #'clusters','gender','cholesterol','gluc','smoke','active','age_group','BMI','map','Diabetes'
        #df['map'] = ((2* df['ap_lo']) + df['ap_hi']) / 3
        # 0   clusters     60142 non-null  uint16
 #1   gender       60142 non-null  int64 
 #2   cholesterol  60142 non-null  int64 
 #3   gluc         60142 non-null  int64 
 #4   smoke        60142 non-null  int64 
 #5   alco         60142 non-null  int64 
 #6   active       60142 non-null  int64 
 #7   cardio       60142 non-null  int64 
# 8   age_group    60142 non-null  int64 
# 9   bmi          60142 non-null  int64 
# 10  map          60142 non-null  int64
        age=int(request.form.get("Age"))
        gender=int(request.form.get("gender"))
        height=int(request.form.get("height"))
        weight=float(request.form.get("weight"))
        ap_hi=float(request.form.get("ap_hi"))
        ap_ho=float(request.form.get("ap_lo"))
        cholesterol=int(request.form.get("cholesterol"))
        
        gluc=int(request.form.get("gluc"))
        smoke=int(request.form.get("smoke"))
        alco=int(request.form.get("alco"))
        active=int(request.form.get("active"))
        Diabetes=int(request.form.get("Diabetes"))
        #bmi, bmi_category = calculate_bmi(weight, height)
        #map_value, map =calculate_map(ap_hi, ap_ho)
       # age_years= float((age / 365.25))
        #age_rounded=(age_years/365.25).round(0).astype(int)
      #  age_rounded = float(round(age_years))
        #print(age,'ageyears',age_years)
        #print(age_rounded)
        # Obtenir le groupe d'âge
        age_group = get_age_group(age)
        
        #print(age_group,' ', gender,' ',cholesterol,' ',gluc,' ',smoke,Diabetes)
        #data=[id,age,gender,height,weight,ap_hi,ap_ho,cholesterol,gluc,smoke,alco,active,age_rounded,age_years]
        # Données à ajouter
        data = [gender, height, weight, ap_hi, ap_ho, cholesterol, gluc, smoke, active, age_group, Diabetes, alco]

        # Chemin du fichier CSV
        file_path = 'test.csv'

        # Utiliser la fonction pour ajouter les données au fichier CSV
        df = ajouter_donnees_au_csv(data, file_path)

        # Afficher le DataFrame mis à jour
       
        # Utilisation de la fonction dans votre application Flask
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

        # Obtenir les valeurs minimales et maximales de BMI
        bmi_min = int(df['BMI'].min())
        bmi_max = int(df['BMI'].max())

        # Discrétiser le BMI en 6 catégories
        df['BMI'] = pd.cut(df['BMI'], bins=6, labels=range(6), right=True, include_lowest=True)

        # Obtenir la distribution des catégories de BMI
        bmi= df["BMI"].value_counts(normalize=True)
        #print(bmi)
        df['map'] = ((2* df['ap_ho']) + df['ap_hi']) / 3

        mapMin = int(df['map'].min())
        mapMax = int(df['map'].max())

       # print(mapMin, mapMax)

        df['map'] = pd.cut(df['map'], bins=6, labels=range(6), right=True, include_lowest=True)
        df_og=df
        df=df.drop(['height','weight','ap_hi','ap_ho'],axis=1)

        

        #df.head()
        from sklearn import  preprocessing
        le = preprocessing.LabelEncoder()
        df = df.apply(le.fit_transform)
        df.describe()

        #df.head()
       # print(map)
        cost = []
        num_clusters = range(1,6) # 1 to 5
        for i in list(num_clusters):
            kmode = KModes(n_clusters=i, init = "Huang", n_init = 5, verbose=0,random_state=1)
            kmode.fit_predict(df)
            cost.append(kmode.cost_)
        km = KModes(n_clusters=2, init = "Huang", n_init = 5,random_state=1)
        clusters = km.fit_predict(df)
        clusters
        df.insert(0,"clusters",clusters,True)
        #print(df)
        

        #df.head()

        #print(df)
        # Récupérer la dernière ligne
        derniere_ligne = df.tail(1)
        # Obtenir la dernière ligne du DataFrame sous forme de dictionnaire
        last_row_dict = df.tail(1).to_dict(orient='records')[0]

        # Construire la structure JSON
        data = {
            "Inputs": {
                "input1": [last_row_dict]
            },
            "GlobalParameters": {}
        }

        # Imprimer les données JSON
        print(data)
        body = str.encode(json.dumps(data))
        url = 'http://4.153.84.236:80/api/v1/service/tall/score'
        # Replace this with the primary/secondary key or AMLToken for the endpoint
        api_key = '3vOt9n24IsQZcNFL9KhlKQdTmq1unhsU'
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")


        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            print(result)
           
       
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))





















        
         #clusters gender	cholesterol	gluc	smoke	alco	active	cardio	age_group	bmi	map
        # Afficher la dernière ligne
        #prediction = predict_model(model,derniere_ligne)
        #model_eval(prediction.prediction_label,y_test)
        #print(prediction)
       
        #if prediction.prediction_label.any()==1 :
          #  output = output +'Malade  '+ '   ' +'precision' + '   ' + str(prediction.prediction_score.iloc[0] * 100 )+ '%'
        #else:
          #  output = output +'non-malade  ' + '    ' + 'precision' + '   ' + str(prediction.prediction_score.iloc[0] * 100 ) + '%'
            
       # return render_template('single_prediction.html',result=output)

    else:
        return render_template('home.html')
       
        #print(derniere_ligne)
        # Si votre tableau est une seule observation, vous pouvez le transformer en une seule ligne de tableau bidimensionnel
        
        
        
      
       







def get_age_group(age):
    # Define the bin edges and labels
    age_edges = [30, 35, 40, 45, 50, 55, 60, 65]
    age_labels = [0, 1, 2, 3, 4, 5, 6]

    # Create the 'age_group' using pd.cut()
    age_group = pd.cut([age], bins=age_edges, labels=age_labels, include_lowest=True, right=True)[0]

    return age_group
 



if __name__=="__main__":
    app.run(host="0.0.0.0")
    app.config['PREFERRED_URL_SCHEME'] = 'https'