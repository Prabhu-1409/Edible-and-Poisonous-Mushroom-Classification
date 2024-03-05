from flask import Flask, render_template, request, url_for, session, redirect
#from waitress import serve
import pandas as pd
import mysql.connector
import re
import hashlib
from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
import numpy as np

model1 = load_model("CNN.h5")


df=pd.read_csv('mushrooms.csv')
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for column in df.columns:
    df[column]=le.fit_transform(df[column])
          
                   
clean_data=df.copy()

a=['cap-shape','cap-surface','cap-color','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
x=clean_data[a].copy()
x.fillna('')
f=pd.DataFrame(x)

b=['class']
y=clean_data[b].copy()
y.fillna('')
clf=DecisionTreeClassifier()
model=clf.fit(x,y)

app = Flask(__name__)


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img)
    img = img.reshape(1, 64, 64, 3)
    img = img / 255.0
    return img

app.secret_key='aa272adf2268a311aef3d0b225abd031d661a66419d23cbe'

connection=mysql.connector.connect(
            host='localhost',
            user='root',
            password='Prabhu@2002',
            database='login'
        )

@app.route('/')
def index():
    return render_template('login.html')


@app.route('/register' ,methods=['GET','POST'])
def register():
    message=''
    if request.method=='POST'and 'firstname' in request.form and 'lastname' in request.form and 'email' in request.form and 'password' in request.form:
        firstname=request.form['firstname']
        lastname=request.form['lastname']
        email=request.form['email']
        password=request.form['password']
        
        cur=connection.cursor()
        cur.execute('SELECT * FROM mushlogin WHERE email= %s',(email,))
        account=cur.fetchone()
        if account:
            message='User Already exist'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+',email):
            message="Invalid Email ID"
        elif not firstname or not lastname or not password:
            message="Please Fill the fields"
        else:
            cur.execute("INSERT INTO mushlogin VALUES (NULL,%s,%s,%s,%s)",(firstname,lastname,email,password,))
            connection.commit()
            return redirect(url_for('login'))
    elif request.method=='POST':
        message ="Fill the details"
    return render_template('register.html',mes=message)

@app.route('/login',methods=['POST','GET'])
def login():
    message=''
    if request.method=='POST' and 'email' in request.form and 'password' in request.form:
        message='started..'
        email=request.form['email']
        password=request.form['password']
        # connection=mysql.connector.connect(host='localhost',user='root',password='Prabhu@2002',database='login')
        cur=connection.cursor()
        cur.execute('SELECT * FROM mushlogin WHERE email = %s',(email,))
        account =cur.fetchone()
        if account:
            stored_password=account[4]
            if(password==stored_password):
                session['loggedin']= True
                session['firstname']=account[1]
                message='Logged In Successfully'
                try:
                    return redirect(url_for('predict'))
                except Exception as e: 
                    message = f'Redirect error: {str(e)}'
            else:
                message='Incorrect'
        else:
            message ='Invalid Email'
    return render_template('login.html', mes=message)

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/classify', methods=['GET','POST'])
def predict():
    result=''
    if request.method=='GET':
        if not session['loggedin']:
            return redirect(url_for('login'))
    in1=request.form.get('capshape')
    in2=request.form.get('capsurface')
    in3=request.form.get('capcolor')
    in6=request.form.get('gillattachment')
    in7=request.form.get('gillspacing')
    in8=request.form.get('gillsize')
    in9=request.form.get('gillcolor')
    in10=request.form.get('stalkshape')
    in11=request.form.get('stalkroot')
    in12=request.form.get('stalksurfaceabovering')
    in13=request.form.get('stalksurfacebelowring')
    in14=request.form.get('stalkcolorabovering')
    in15=request.form.get('stalkcolorbelowring')
    in16=request.form.get('veiltype')
    in17=request.form.get('veilcolor')
    in18=request.form.get('ringnumber')
    in19=request.form.get('ringtype')
    in20=request.form.get('sporeprint')
    in21=request.form.get('population')
    in22=request.form.get('habitat')
    input_features=[in1,in2,in3,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16,in17,in18,in19,in20,in21,in22]
    try:
        prediction = model.predict([input_features])
        if prediction[0]==0:
            result='Edible'
        elif prediction[0]==1:
            result='Poisonous'
        else:
            result='Invalid'
        return render_template('classify.html', output=result)
    except Exception as e:
        return render_template('classify.html', output=f'Error: {str(e)}')

@app.route('/image', methods=['GET', 'POST'])
def image():
    result = None
    uploaded_file = None

    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            img_path = f"static/{uploaded_file.filename}"
            uploaded_file.save(img_path)
            

            img = preprocess_image(img_path)
            prediction = model1.predict(img)
            if prediction[0][0] < 0.5:
                result = "Edible"
            else:
                result = "Poisonous"

    return render_template('cnn.html', result=result)


if __name__ == "__main__":
    #serve(app, host="0.0.0.0", port=5000)
    app.run()
