from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained KMeans model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Expected columns for prediction
EXPECTED_COLUMNS = ['Age', 'Gender', 'Musical Experience', 
                    'Experience Volunteering with Beneficiary', 
                    'Past Volunteer?', 'Role']

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)

        # Ensure all expected columns are present
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0  # Default to 0 for missing columns

        # Reorder columns to match the trained model's structure
        df = df[EXPECTED_COLUMNS]

        # Predict clusters using the pre-trained model
        labels = kmeans.predict(df)

        # Return cluster labels
        return jsonify({"clusters": labels.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

