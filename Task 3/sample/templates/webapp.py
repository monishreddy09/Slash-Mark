from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = "samples"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        files = request.files.getlist("files")

        for file in files:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

        os.system("python app.py --input-dir samples --threshold 0.65")
        return "Plagiarism Check Completed! Check outputs folder."

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)