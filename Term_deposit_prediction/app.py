from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('/Users/aplle/Desktop/Streamlit/Term_deposit_prediction/model/rf_Classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a route for serving the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions    
@app.route('/submit-form', methods=['POST','GET'])
def submit_form():    
    print("Form data:", request.form)
    # Remaining code
    age = request.form['age']    
    job = request.form['job']    
    marital = request.form['marital']     
    education = request.form['education']    
    default_input = request.form['default-input']     
    balance = request.form['balance']     
    housing = request.form['housing']     
    loan = request.form['loan']    
    day = request.form['day']     
    month = request.form.get('month')   # Adjusted line
    duration = request.form['duration']    
    campaign = request.form['campaign']    
    pdays = request.form['pdays']    
    previous = request.form['previous']     
    print(job)
     # Perform further processing with the retrieved form datareturn 'Form submitted successfully'
    return 'Form submitted successfully'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the JSON payload
    data = request.get_json()
    print("Received data:", data) 
    data2 = request.form['job']  # Adjusted line
    print(data2)
    # Preprocess categorical variables using one-hot encoding
    categorical_features = {
        'job': ['blue-collar', 'technician', 'management', 'admin.', 'services', 'entrepreneur', 'retired', 'self-employed', 'unemployed', 'student', 'housemaid'],
        'marital': ['married', 'single', 'divorced'],
        'education': ['secondary', 'tertiary', 'primary'],
        'default': ['no', 'yes'],
        'housing': ['no', 'yes'],
        'loan': ['no', 'yes'],
        'month': ['may', 'june'],
        
    }
    """ Select categorical columns for one-hot encoding
    cat_cols = ['marital','job', 'education', 'default', 'housing', 'loan', 'month','Target']

     Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    Display the first few rows of the encoded DataFrame
    print(df_encoded.head())"""


    for feature, categories in categorical_features.items():
        for category in categories:
            encoded = 1 if data[feature] == category else 0
            data[f'{feature}_{category}'] = encoded
        # Remove the original categorical feature from the data
        del data[feature]

    # Convert data to numpy array and reshape if necessary
    data = np.array(list(data.values())).reshape(1, -1)

    # Perform predictions
    prediction = model.predict(data)

    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
