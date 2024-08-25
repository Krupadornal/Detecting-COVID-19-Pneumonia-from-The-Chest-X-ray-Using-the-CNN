import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

app = Flask(__name__)
model = load_model('models/trained_model.h5')
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0][0]
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        os.remove(file_path)
        if int(preds) == 1:return "Patient is infected by covid 19"
        else:return "Patient is Normal"

if __name__ == '__main__':
    app.run()