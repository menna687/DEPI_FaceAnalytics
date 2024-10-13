from flask import Flask, request, jsonify
from flask_cors import CORS
from model import FaceAnalysisModel

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (needed if frontend runs on a different domain)

# Load the face analysis model, including the dataset
model = FaceAnalysisModel()

@app.route('/dataset-info', methods=['GET'])
def get_dataset_info():
    # Return some basic information about the dataset (columns, shape, etc.)
    dataset_info = model.get_dataset_info()
    return jsonify(dataset_info), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        anchor_image = data.get('anchor')
        positive_image = data.get('positive')
        negative_image = data.get('negative')

        # Ensure all required images are provided
        if not anchor_image or not positive_image or not negative_image:
            return jsonify({'error': 'Missing image data'}), 400

        # Use the model to make predictions, possibly involving the dataset
        result = model.predict(anchor_image, positive_image, negative_image)
        
        # Return the result as a JSON response
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
