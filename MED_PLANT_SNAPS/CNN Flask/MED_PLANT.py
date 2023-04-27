import numpy as np
import os
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("C:\AI HACK\MODEL\MED_PLANT_SNAP(19.03.2023).h5")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(150, 150, 3))
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x, axis=0)
        print(x)
        preds = model.predict(x)
        print(preds)
        pred = np.argmax(preds[0])
        index = ["Arive-Dantu", "Basale", "Betel", "Crape_Jasmine", "Curry", "Drumstick", "Fenugreek", "Guava", "Hibiscus", "Indian_Beech",
                 "Indian_Mustard", "Jackfruit", "Jamaica_Cherry-Gasagase", "Jamun", "Jasmine", "Karanda", "Lemon",
                 "Mango", "Mexican_Mint", "Mint", "Neem", "Oleander", "Parijata", "Peepal", "Pomegranate", "Rasna", "Rose_apple",
                 "Roxburgh_fig", "Sandalwood", "Tulsi"]
        class_prediction=index[pred]
        df = pd.read_csv("C:\AI HACK\details.csv")
        def answer(class_prediction):
            for i in range(0, 29):
                if class_prediction == df.iloc[i, 0]:
                    ans1="\nThe plant that you have uploaded is : \n"+ df.iloc[i, 0]+"\nScientific Name : \n"+ df.iloc[i, 1]+"\nDisease Treated :\n "+df.iloc[i, 2]+"\nPreparation Method : \n"+df.iloc[i, 3]+"\nAdministration : \n"+ df.iloc[i, 4]
            return ans1
        print(answer(class_prediction))
        text = answer(class_prediction)
    return text


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
