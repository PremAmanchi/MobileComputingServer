import os
import datetime
from base64 import b64decode
from flask import Flask, request, make_response
from img_to_mnist import classify

UPLOAD_FOLDER = 'static/images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/upload-image", methods=["POST"])
def upload_image():
    json_obj = request.json
    encoded_img = json_obj['encoded_image']
    with open(f"/tmp/img", "wb") as f:
        f.write(b64decode(encoded_img.replace("%20", "\n")))
    category = str(classify("/tmp/img"))
    if encoded_img == '' or category == '':
        return "Invalid image or category", 500
    else:
        filename = f'{category}-{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}'
        loc = os.path.join(app.config['UPLOAD_FOLDER'], category)
        if not os.path.isdir(loc):
            os.mkdir(loc)
        with open(f"{loc}/{filename}", "wb") as f:
            f.write(b64decode(encoded_img.replace("%20", "\n")))
        return "Success", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
