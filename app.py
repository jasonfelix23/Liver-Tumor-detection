from flask import Flask, render_template, request, redirect, jsonify, url_for, flash, session
import sqlite3
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)
db_local = 'patients.db'
model = keras.models.load_model("classification.h5")


UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

result = ""
ses = False

#  ============================================== MODEL ==============================================

def check(number, filename):
    if number == 0:
        return 'Tumor Not Detected'
    elif number==2:
        return 'Tumor Detected'
    elif number==3:
        return 'Tumor Detected'
    elif number==4:
        return 'Tumor Detected'
    


def process_img(img, add_pixels_value=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Thresholding the image, and performing a series of erosions + dilations to remove any regions of noise
    thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    new_img = cv2.dilate(thresh, None, iterations=2)

    return np.array(new_img, dtype=object)


@app.route("/mainPage", methods=["GET", "POST"])

def mainPage():

    if request.method == 'POST':
        file = request.files['mri']
        if 'mri' not in request.files:
            return render_template('btd.html', ses=ses, error="File not found!!!")

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('btd.html', ses=ses, error="File not found!!!")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'work.jpg')
            file.save(filepath)
            img = cv2.imread("./static/trial.jpg")
            img_array = process_img(img)
            cv2.imwrite("assets/new/5.jpg", np.float32(img_array))
            img_get = cv2.imread("assets/new/5.jpg")
            img_fin = cv2.resize(img_get, (150, 150))
            img_array = np.array(img_fin)
            img_batch = np.expand_dims(img_array, axis=0)
            img_array = img_array.reshape(1, 150, 150, 3)
            prediction = model.predict_on_batch(img_array)
            classification = np.where(prediction == np.amax(prediction))[1][0]
            predicted_results = check(classification, filename)
            result = predicted_results
            print(result)
            return render_template("result.html", img=img_fin, filename=filename, predicted_results=predicted_results, ses=ses, error="")
    return render_template("btd.html", ses=ses, error="")


@app.route("/result")
def result():
    return render_template('result.html', predicted_result="From the outer", ses=ses, error="")


#  ============================================== form data backend ==============================================


@ app.route("/form",  methods=["GET", "POST"])
def form():

    if request.method == "POST":
        user_details = (
            request.form['name'],
            request.form['age'],
            request.form['gender'],
            request.form['bgrp'],
            request.form['mHist'],
            request.form['pNo'],
            request.form['tdate'],
            request.form['report']
        )
        print(user_details)
        insertdata(user_details)
        return redirect(url_for('displayData'))
    return render_template('info.html', ses=ses)


def insertdata(user_details):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    sql_execute_string = 'INSERT INTO pInfo(pname, page, pgender, pbgrp, pmedhist, pphone, pdate, presult) VALUES (?,?,?,?,?,?,?,?)'
    c.execute(sql_execute_string, user_details)
    conn.commit()
    conn.close()
    print(user_details)


def query_data():
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    c.execute("""
       SELECT * 
    FROM    pInfo
    WHERE   id = (SELECT MAX(id)  FROM pInfo);

    """)
    user_data = c.fetchall()
    return user_data


@ app.route("/displayData", methods=["GET", "POST"])
def displayData():
    if request.method == "GET":
        user_data = query_data()
        print(user_data)
        return render_template('display.html', user_data=user_data, ses=ses)

#  ============================================== Login/Sign Up Backend ==============================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":
        conn = sqlite3.connect(db_local)
        c = conn.cursor()
        username = request.form['username']
        password = request.form['password']
        fullname = request.form['fullname']
        emailid = request.form['emailid']
        hname = request.form['hname']
        position = request.form['position']

        c.execute("SELECT * FROM logindb WHERE username = ?", [username])
        if c.fetchone() is not None:
            return render_template('signup.html', error="Username already taken")
        elif(checkpass(password) == True):
            c.execute(
                "INSERT INTO logindb(username, password, fullname, emailid, hname, position ) VALUES (?,?,?,?,?,?)", (username, password, fullname, emailid, hname, position))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        else:
            return render_template('signup.html', ses=ses, error="The password is weak, use another!!!")
        conn.commit()
        conn.close()
        return render_template('signup.html', ses=ses, error="")
    else:
        return render_template('signup.html', ses=ses, error="")


def checkpass(password):
    Special = ['$', '@', '#']
    if len(password) < 8 or len(password) > 15:
        return False
    if not any(char.isdigit() for char in password):
        return False
    if not any(char.isupper() for char in password):
        return False
    if not any(char.islower() for char in password):
        return False
    if not any(char in Special for char in password):
        return False
    return True


@ app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = sqlite3.connect(db_local)
        c = conn.cursor()
        username = request.form['username']
        password = request.form['password']
        c.execute("select * from logindb where username=? and password =?",
                  (username, password))
        row = c.fetchone()
        if row == None:
            return render_template('login.html', error="Login Failed: No such user exists")
        else:
            global ses
            ses = True
            return render_template('btd.html', ses=ses, error="")

    return render_template("login.html", error="")


@ app.route("/logout")
def logout():
    global ses
    ses = False

    return render_template('btd.html', ses=ses, error='')


#================================== DOCTOR'S PAGE ==============================================

@ app.route("/doctors/<city>")
def doctors(city):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    city = str(city)
    c.execute("select * from doctors where city=? ",
              (city,))
    data = c.fetchall()
    conn.close()
    return render_template('doctors.html', data=data, error="")


if __name__ == '__main__':

    app.run(debug=True)
