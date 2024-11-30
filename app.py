from flask import Flask, request, render_template
import numpy as np
import os
import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt

app = Flask(__name__)

# Ensure the upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set paths for your files
target_shape = (75, 75)  # Target shape for model input
saved_model_dir = 'files'  # Directory where your model is saved

# Custom Layer Definition
@tf.keras.utils.register_keras_serializable()
class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, test):
        return tf.reduce_sum(tf.square(anchor - test), -1)

# Load the complete model with custom DistanceLayer
complete_model = tf.keras.models.load_model(os.path.join(saved_model_dir, 'model_epoch_30.keras'))

# Remove the negative image branch from the model
inference_model = Model(complete_model.inputs[:2], complete_model.outputs[:3])

def inference_preprocess(image):
    """
    Preprocess the input image by resizing it to the target shape.
    """
    image = tf.image.resize(image, target_shape)  # Resize the image
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
    return image  # Return the preprocessed image

@app.route('/', methods=['GET', 'POST'])
def index():
    gender_label = age_label = verification = ""
    anchor_image_path = test_image_path = None  # Initialize paths

    if request.method == 'POST':
        # Handle the uploaded files (anchor and test images)
        anchor_file = request.files['anchor_image']
        test_file = request.files['test_image']

        if anchor_file and test_file:
            # Load the images directly from uploaded files
            anchor_images = np.load(anchor_file)  # Load all anchor images
            test_images = np.load(test_file)      # Load all test images
            
            # Choose random images from the loaded arrays
            random_index_anchor = np.random.randint(0, anchor_images.shape[0])  # Random index for anchor images
            random_index_test = np.random.randint(0, test_images.shape[0])      # Random index for test images
            
            # Select random anchor and test images
            anchor_image = anchor_images[random_index_anchor]
            test_image_neg = test_images[random_index_test]

            # Save the anchor and test images to display later
            anchor_image_path = os.path.join(UPLOAD_FOLDER, 'anchor_image.png')
            test_image_path = os.path.join(UPLOAD_FOLDER, 'test_image.png')

            plt.imsave(anchor_image_path, anchor_image)
            plt.imsave(test_image_path, test_image_neg)

            # Preprocess images
            anchor_image = inference_preprocess(anchor_image)
            test_image_neg = inference_preprocess(test_image_neg)

            # Add batch dimension (for model input)
            anchor_image = np.expand_dims(anchor_image, axis=0)
            test_image_neg = np.expand_dims(test_image_neg, axis=0)

            # Perform inference
            gender, age, distance = inference_model([anchor_image, test_image_neg])

            # Determine age label based on model output
            age_threshold = 15  # Define the threshold for young

            # if isinstance(age, float):  # If age is a continuous value
            #     age_label = 'Young' if age < age_threshold else 'Not young'
            # else:  # If age is a probability score
            age_label = 'young' if age > 0.5 else 'Old'

            # Assign labels based on model output
            gender_label = 'Male' if gender > 0.5 else 'Female'
            verification = 'Negative' if distance > 2.5 else 'Positive'

    return render_template('index.html', gender=gender_label, age=age_label, verification=verification,
                           anchor_image_path=anchor_image_path, test_image_path=test_image_path)

if __name__ == '__main__':
    app.run(debug=True)
