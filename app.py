import flask
import pickle
import pandas as pd

# Load the machine learning model
with open('model/InsuranceModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = flask.Flask(__name__, template_folder='templates')

# Define the main route
@app.route('/', methods=["GET", "POST"])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == "POST":
        # Get input data from the form
        age = flask.request.form['age']
        bmi = flask.request.form['BMI']
        children = flask.request.form['Children']

        # Create a DataFrame with input data
        input_variables = pd.DataFrame([[age, bmi, children]],
                                       columns=['age', 'bmi', 'children'])

        # Make prediction using the model
        prediction = model.predict(input_variables)[0]

        # Render the result template with original input and prediction
        return flask.render_template('result.html',
                                     original_input={'Age': age,
                                                     'BMI': bmi,
                                                     'Children': children},
                                     result=prediction)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)
