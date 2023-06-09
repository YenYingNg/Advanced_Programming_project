# Importing required libraries
from flask import Flask, render_template, request
from model import predict_result
import base64

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
            
            img_data = request.files['file'].read()
            img_data = base64.b64encode(img_data).decode('utf-8')
            
            img_stream = request.files['file'].stream      
            pred = predict_result(img_stream)

            return render_template("result.html", predictions=str(pred), img_data=img_data)

    except Exception as e:
        msg = "error: "+str(e)
        return render_template("result.html", err=msg)



# Driver code
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=True)
