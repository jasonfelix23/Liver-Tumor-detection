import sre_compile
from flask import Flask, render_template, request, redirect, jsonify, url_for, flash, session
import sqlite3
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plot
import nibabel as nib
from PIL import Image
import os

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *


app = Flask(__name__)
db_local = 'patients.db'


path = Path(os.getcwd())
def get_x(fname:Path): return fname
def label_func(x): return path/'train_masks'/f'{x.stem}_mask.png'
def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean() 

def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0


learn0 = load_learner(path/f'Liver_segmentation_unet.h5',cpu=False )
#model = keras.models.load_model("Liver_segmentation_unet.h5")


UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

result = ""
ses = False
name = ""

#  ============================================== MODEL ==============================================

def check(number, filename):
    
    if number == 0:
        return 'Tumor Not Detected'
    elif number==1:
        return 'Tumor Not Detected'
    else:
        return 'Tumor Detected'

def malignantBeningCheck(count):
    if(count <500):
        return 'malignant'
    else:
        return 'Benign'
    


def process_img(img, add_pixels_value=0):
    new_img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

    return np.array(new_img, dtype=object)

def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom = (200,60)
)

@patch
def windowed(self:Tensor, w, l):
    px = self.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

class TensorCTScan(TensorImageBW): _show_args = {'cmap':'bone'}

@patch
def freqhist_bins(self:Tensor, n_bins=100):
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()
    
@patch
def hist_scaled(self:Tensor, brks=None):
    if self.device.type=='cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.,1.)
    
    
@patch
def to_nchan(x:Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins,int) or bins!=0: res.append(x.hist_scaled(bins).clamp(0,1))
    dim = [0,1][x.dim()==3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def save_jpg(x:(Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins)*255).byte()
    im = Image.fromarray(x.permute(1,2,0).numpy(), mode=['RGB','CMYK'][x.shape[0]==4])
    im.save(fn, quality=quality)

def processNiifiles(filepath):
    curr_ct        = read_nii("static/"+filepath)
    curr_file_name = str(filepath).split('.')[0]
    curr_dim       = curr_ct.shape[2] # 512, 512, curr_dim
    curr_count = 0
    file_array = []
    for curr_slice in range(int(curr_dim/2),curr_dim,3): 
        data = tensor(curr_ct[...,curr_slice].astype(np.float32))
        curr_count += 1
        save_file_name = f'{curr_file_name}_slice_{curr_count}.jpg'
        data.save_jpg(f"static/"+save_file_name, [dicom_windows.liver,dicom_windows.custom])
        file_array.append(save_file_name)
        if(curr_count == 4):
            break;
    return file_array

@app.route("/nii", methods=["GET", "POST"])
def nii():
        

    if request.method == 'POST':
        for filename in os.listdir('static/'):
            if filename.startswith('work'):  # not to remove other images
                os.remove('static/' + filename)
            if filename.startswith('scan'):
                os.remove('static/' + filename)
            if filename.startswith('pred'):
                os.remove('static/' + filename)
            if filename.startswith('slice'):
                os.remove('static/' + filename)
                
        file = request.files['nii']
        if 'nii' not in request.files:
            return render_template('nii.html', ses=ses, error="File not found!!!")

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('nii.html', ses=ses, error="File not found!!!")
        if file:
            filename = secure_filename(file.filename)
            timeStamp ="work" + str(time.time()) + ".nii"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timeStamp )
            
            file.save(filepath)
            file_array = processNiifiles(timeStamp)

            #print(prediction)
            return render_template("imagesNii.html",file_array = file_array, ses=ses,name=name, error="")
    return render_template("nii.html",name= name, ses=ses, error="")

@app.route("/imagesNii", methods=['GET', 'POST'])
def imagesNii():
    return render_template('imagesNii.html', ses=ses, error="")

@app.route("/predNii/<id>")
def predNii(id):
    for filename in os.listdir('static/'):
        if f'slice_{id}' in filename:
            timeStamp = filename
            break
    
    img = cv2.imread("./static/"+timeStamp)
    img_array = process_img(img)
    img_path0 ="scan" + str(time.time()) + ".jpg"
    img_path = "static/"+img_path0
    cv2.imwrite(img_path, np.float32(img_array))
    img_get = cv2.imread(img_path)
    img_fin = cv2.resize(img_get, (512, 512))
    img_array = np.array(img_fin)
    print(img_array.shape)
    img_batch = np.expand_dims(img_array, axis=0)
    img_array = img_array.reshape(-1, 512, 512, 3)
    test_files = [img]
    test_dl = learn0.dls.test_dl(test_files)
    preds, y = learn0.get_preds(dl=test_dl)

    predicted_mask = np.argmax(preds, axis=1)
    pred_path ="pred" + str(time.time()) + ".jpg"

    plt.imsave('static/'+pred_path,predicted_mask[0])
    #prediction = learn0.predict(img_fin)
    #predicted_mask = np.argmax(prediction, axis=1)
    a=np.array(predicted_mask[0])
    print(np.amin(a))
    classification = np.amax(a)
    #classification = np.where(prediction == np.amax(prediction))[1][0]
    #print(classification)
    predicted_results = check(classification, filename)
    result = 1

    unique, counts = np.unique(a, return_counts =True)
    pred_matrix = np.array((unique, counts)).T
    print( pred_matrix)
    if 1 in unique:
        liver_visiblity = pred_matrix[1][1]/ (pred_matrix[0][1]+pred_matrix[1][1])*250
        liver_visiblity = float("{:.2f}".format(liver_visiblity))
    else:
        liver_visiblity = 0

    if 2 in unique:
        size_result = malignantBeningCheck(counts[2])
        liver_tumor_ratio = pred_matrix[2][1]/ pred_matrix[1][1]
        liver_tumor_ratio = float("{:.3f}".format(liver_tumor_ratio))
        print(size_result)
    else:
        liver_tumor_ratio = "-"
        size_result = "-"

    global ltr,lv, sr
    ltr = liver_tumor_ratio
    lv = liver_visiblity
    sr = size_result



            #print(prediction)
    return render_template("result.html", img1 = timeStamp, img2 = img_path0, img3 = pred_path, predicted_results= predicted_results, size_result = size_result,
            liver_visiblity= liver_visiblity, liver_tumor_ratio = liver_tumor_ratio,ses=ses,name=name, error="")
    


@app.route("/mainPage", methods=["GET", "POST"])

def mainPage():
        

    if request.method == 'POST':
        for filename in os.listdir('static/'):
            if filename.startswith('work'):  # not to remove other images
                os.remove('static/' + filename)
            if filename.startswith('scan'):
                os.remove('static/' + filename)
            if filename.startswith('pred'):
                os.remove('static/' + filename)
                
        file = request.files['mri']
        if 'mri' not in request.files:
            return render_template('index.html', ses=ses, error="File not found!!!")

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('index.html', ses=ses, error="File not found!!!")
        if file:
            filename = secure_filename(file.filename)
            timeStamp ="work" + str(time.time()) + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timeStamp )
            
            file.save(filepath)
            img = cv2.imread("./static/"+timeStamp)
            img_array = process_img(img)
            img_path0 ="scan" + str(time.time()) + ".jpg"
            img_path = "static/"+img_path0
            cv2.imwrite(img_path, np.float32(img_array))
            img_get = cv2.imread(img_path)
            img_fin = cv2.resize(img_get, (512, 512))
            img_array = np.array(img_fin)
            print(img_array.shape)
            img_batch = np.expand_dims(img_array, axis=0)
            img_array = img_array.reshape(-1, 512, 512, 3)
            test_files = [img]
            test_dl = learn0.dls.test_dl(test_files)
            preds, y = learn0.get_preds(dl=test_dl)

            predicted_mask = np.argmax(preds, axis=1)
            pred_path ="pred" + str(time.time()) + ".jpg"

            plt.imsave('static/'+pred_path,predicted_mask[0])
            #prediction = learn0.predict(img_fin)
            #predicted_mask = np.argmax(prediction, axis=1)
            a=np.array(predicted_mask[0])
            print(np.amin(a))
            classification = np.amax(a)
            #classification = np.where(prediction == np.amax(prediction))[1][0]
            #print(classification)
            predicted_results = check(classification, filename)
            result = 1
            unique, counts = np.unique(a, return_counts =True)
            pred_matrix = np.array((unique, counts)).T
            print( pred_matrix)
            if 1 in unique:
                liver_visiblity = pred_matrix[1][1]/ (pred_matrix[0][1]+pred_matrix[1][1])*250
                liver_visiblity = float("{:.2f}".format(liver_visiblity))
            else:
                liver_visiblity = 0

            if 2 in unique:
                size_result = malignantBeningCheck(counts[2])
                liver_tumor_ratio = pred_matrix[2][1]/ pred_matrix[1][1]
                liver_tumor_ratio = float("{:.3f}".format(liver_tumor_ratio))
                print(size_result)
            else:
                liver_tumor_ratio = "-"
                size_result = "-"
                    
            global ltr,lv, sr
            ltr = liver_tumor_ratio
            lv = liver_visiblity
            sr = size_result

            #print(prediction)
            return render_template("result.html", img1 = timeStamp, img2 = img_path0, img3 = pred_path, predicted_results= predicted_results, size_result = size_result,
             liver_visiblity =liver_visiblity,liver_tumor_ratio =liver_tumor_ratio,ses=ses,name=name, error="")
    return render_template("index.html",name= name, ses=ses, error="")



@app.route("/result", methods=['GET', 'POST'])
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
        print(name)
        user_data = query_data()
        dr_data = query_dr_data(name)
        
        print(user_data)
        print(dr_data)
        return render_template('display.html',user_data=user_data,dr_data = dr_data, ltr =ltr, lv=lv, sr=sr, ses=ses, name=name)
    return render_template('info.html', ses=ses, name =name)


def insertdata(user_details):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    sql_execute_string = 'INSERT INTO pInfo(pname, page, pgender, pbgrp, pmedhist, pphone, pdate, presult) VALUES (?,?,?,?,?,?,?,?)'
    c.execute(sql_execute_string, user_details)
    conn.commit()
    conn.close()
    print(user_details)


def query_dr_data(name):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    c.execute("select * from logindb where username=?",
                  (name,))
    dr_data = c.fetchall()
    return dr_data


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
        print(name)
        user_data = query_data()
        dr_data = query_dr_data(name)
        
        print(user_data)
        print(dr_data)
        return render_template('display.html',user_data=user_data,dr_data = dr_data, ltr =ltr, lv=lv, sr=sr, ses=ses, name=name)

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
            global name
            ses = True
            name = row[0]
            return render_template('index.html',name = name, ses=ses, error="")

    return render_template("login.html", error="")


@ app.route("/logout")
def logout():
    global ses
    ses = False
    name = ''

    return render_template('login.html', ses=ses, name=name, error='')


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
