import pickle
from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb')as file:
    clf = pickle.load(file)

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        myDict = request.form
        fever = int(myDict['fever'])
        pain = int(myDict['pain'])
        age = int(myDict['age'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        return render_template('show.html', inf = round(infProb * 100))
    return render_template('index.html')

@app.route("/about", methods = ["GET"])
def about():
    return render_template("about.html")

@app.route("/contact", methods = ["GET"])
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)