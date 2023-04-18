from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the CSV file from a local path into a pandas dataframe
data = pd.read_csv('./weatherHistory.csv', low_memory=False)

# Split data into features and target
X = data[['Temperature', 'Apparent Temperature', 'Humidity', 'Wind Speed', 'Wind Bearing', 'Visibility', 'Pressure']]
y = data['Summary']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
print("Started")

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        temperature = float(request.form['temperature'])
        apparent_temperature = float(request.form['apparent_temperature'])
        humidity = float(request.form['humidity']) / 100
        wind_speed = float(request.form['wind_speed'])
        wind_bearing = float(request.form['wind_bearing'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])

        # Make prediction
        new_data = pd.DataFrame({
            'Temperature': [temperature],
            'Apparent Temperature': [apparent_temperature],
            'Humidity': [humidity],
            'Wind Speed': [wind_speed],
            'Wind Bearing': [wind_bearing],
            'Visibility': [visibility],
            'Pressure': [pressure]
        })
        prediction = model.predict(new_data)

        # Display prediction
        return render_template('prediction.html', prediction=prediction[0])
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
