# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
from PIL import Image


# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img_stream = request.files['file'].stream
            tensor_img = preprocess_img(img_stream)
            pred = predict_result(tensor_img)
            return render_template("result.html", predictions=str(pred))

    except Exception as e:
        msg = "error: "+str(e)
        return render_template("result.html", err=msg)



def image_info(img):
    # img = Image.open(img_stream)
            
    # Get basic image information
    image_format = img.format
    image_size = img.size
            
    # Construct a string with the image information
    info_str = f"Format: {image_format}<br>"
    info_str += f"Size: {image_size}<br>"
    return info_str


# Driver code
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=True)
