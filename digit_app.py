
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io, base64

app = Flask(__name__)
model = load_model("handwritten_digit_model.h5")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognizer</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 30px; }
        canvas { border: 2px solid #000; }
        #result { font-size: 24px; margin-top: 20px; }
    </style>
</head>
<body>
    <h2>Draw or Upload a Digit</h2>
    <canvas id="canvas" width="280" height="280"></canvas><br>
    <button onclick="clearCanvas()">Clear</button>
    <button onclick="predictCanvas()">Predict</button><br><br>
    <input type="file" id="imageUpload" accept="image/*"><br><br>
    <button onclick="predictUpload()">Predict Uploaded Image</button>
    <h3 id="result">Prediction: </h3>

    <script>
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        canvas.isDrawing = false;
        canvas.addEventListener("mousedown", () => { canvas.isDrawing = true; });
        canvas.addEventListener("mouseup", () => { canvas.isDrawing = false; ctx.beginPath(); });
        canvas.addEventListener("mousemove", draw);

        function draw(e) {
            if (!canvas.isDrawing) return;
            ctx.lineWidth = 15;
            ctx.lineCap = "round";
            ctx.strokeStyle = "black";
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
        }

        function clearCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById("result").innerText = "Prediction: ";
        }

        function predictCanvas() {
            let imageData = canvas.toDataURL("image/png");
            fetch("/predict", {
                method: "POST",
                body: new URLSearchParams({ imageData })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("result").innerText = "Prediction: " + data.prediction;
            });
        }

        function predictUpload() {
            let fileInput = document.getElementById("imageUpload");
            if (!fileInput.files[0]) return;
            let formData = new FormData();
            formData.append("image", fileInput.files[0]);

            fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("result").innerText = "Prediction: " + data.prediction;
            });
        }
    </script>
</body>
</html>
"""

def prepare_image(image):
    image = ImageOps.invert(image.convert("L")).resize((28, 28))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 28, 28)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' in request.files:
        image = Image.open(request.files['image'].stream)
    else:
        data_url = request.form['imageData']
        content = data_url.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(content)))

    processed_image = prepare_image(image)
    prediction = np.argmax(model.predict(processed_image), axis=-1)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
