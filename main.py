import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the CSV file from a local path into a pandas dataframe
data = pd.read_csv('./weatherHistory.csv', low_memory=False)


# Split data into features and target
X = data[['Temperature', 'Apparent Temperature', 'Humidity', 'Wind Speed', 'Wind Bearing', 'Visibility', 'Pressure']]
y = data['Summary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Test model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100}%")

# Take User Input
print('Please enter the current weather information:')
temperature = float(input('Temperature (in degrees Celsius): '))
apparent_temperature = float(input('Apparent Temperature (in degrees Celsius): '))
humidity = float(input('Humidity (as a percentage): '))
wind_speed = float(input('Wind speed (in meters per second): '))
wind_bearing = float(input('Wind bearing (in degrees): '))
visibility = float(input('Visibility (in kilometers): '))
pressure = float(input('Pressure (in millibars): '))


# Make prediction
new_data = pd.DataFrame({
    'Temperature': [temperature],
    'Apparent Temperature': [apparent_temperature],
    'Humidity': [humidity / 100],
    'Wind Speed': [wind_speed],
    'Wind Bearing': [wind_bearing],
    'Visibility': [visibility],
    'Pressure': [pressure]
})


prediction = model.predict(new_data)
print(f"Prediction: {prediction[0]}")
