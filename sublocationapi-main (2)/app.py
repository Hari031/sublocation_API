import io
import flask
import requests
from PIL import Image 
from flask import request, jsonify

import modellib.model_inference as model
from modellib.blobclient import upload_to_blob

model_path = r"models/vm_1620796160.pt"
voc_txt_path = r"models/vm_1620796160_voc-model-labels_woffrr_10th.txt"
model_path_fr = r"models/vm_1637600421 (1).pt"
voc_txt_path_fr= r"models/voc-model-labels (1).txt"
location_model = r"models/location.pt"
location_txt = r"models/location-labels.txt"

app = flask.Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    
    files = request.files.getlist('files[]')
    image_url = str(request.form.get("image_url"))
    
    file_urls = []

    if len(image_url) > 0 and image_url != 'None':
        with Image.open(requests.get(image_url, stream=True).raw) as img:
            buf = io.BytesIO()                        
            img.convert('RGB').save(buf, format="png") 
            image_name = image_url.rsplit('/', 1)[1]
            img_link = upload_to_blob(image_name,buf.getvalue())
            file_urls.append(img_link)
            
    
    if (len(files)) > 0 and (str(files) != "[<FileStorage: '' (None)>]"):
        for file in files:
            with Image.open(file) as img:
                buf = io.BytesIO()                        
                img.convert('RGB').save(buf, format="png") 
                image_name = file.filename
                img_link = upload_to_blob(image_name,buf.getvalue())
                file_urls.append(img_link)
                
  
    print('file_urls',file_urls)
    dict1 = model.driver(file_urls, model_path, voc_txt_path, model_path_fr, voc_txt_path_fr,location_model,location_txt)
    
    
   
    return jsonify(dict1)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)