from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("outputs/cat_dog_cnn_model.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)

            if pred[0][0] > 0.5:
                prediction = "Dog"
            else:
                prediction = "Cat"

            return render_template("index.html",
                                   prediction=prediction,
                                   image_path=filepath)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)