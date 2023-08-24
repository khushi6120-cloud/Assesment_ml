

#1) Data preprocessing and cleanin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_excel("customer_churn_large_dataset.xlsx")
data= data.drop(columns=['CustomerID', 'Name'])

# Initial data exploration
print(data.head())

# Handle missing data
data.dropna(inplace=True)

# Handle outliers (optional)

# Encode categorical variables
label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])
data["Location"] = label_encoder.fit_transform(data["Location"])

# Splitting data
X = data.drop("Churn", axis=1)
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 2)Feature scaling or normalization (if needed)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# 3)Model selection and optimization.
# i have choose two model one is Logistic Regression and second is ANN
# Choose a model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Validate the model
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_class = [1 if pred > 0.5 else 0 for pred in y_pred]

accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1_score:", f1)





# 4) fine-tune hyperparameters and explore cross-validation here
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_


# 5)Model deployment and integration.

#create a simple function using Flask or FastAPI to simulate deployment



import tensorflow as tf
import numpy as np

# ... Train and build your model ...

# Save the trained model in h5 format
model.save("trained_model.h5")

# Save the scaler
np.savez("scaler.npz", scaler=scaler,allow_pickle=True)

# Load the scaler (used during training)


from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained neural network model
model = tf.keras.models.load_model("trained_model.h5")

# Load the scaler (used during training)
scaler_data = np.load("scaler.npz", allow_pickle=True)
scaler = scaler_data['scaler']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [data['Age'], data['Gender'], data['Location'], data['Subscription_Length_Months'],
                    data['Monthly_Bill'], data['Total_Usage_GB']]

        # Apply the same preprocessing as during training
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)
        churn_probability = prediction[0][0]

        if churn_probability >= 0.5:
            result = "Churn"
        else:
            result = "No Churn"

        return jsonify({"prediction": result, "churn_probability": float(churn_probability)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
#
#
# # In[ ]:
#



# Ensure the model can take new customer data as input and provide churn predictions.

import requests

url = "http://127.0.0.1:5000/predict"

input_data = {
    "Age": 30,
    "Gender": 1,
    "Location": 2,
    "Subscription_Length_Months": 12,
    "Monthly_Bill": 80,
    "Total_Usage_GB": 200
}

response = requests.post(url, json=input_data)
prediction = response.json()

print("Churn Prediction:", prediction["churn_prediction"])
# Save the trained model in native Keras format
model.save("trained_model.h5")



