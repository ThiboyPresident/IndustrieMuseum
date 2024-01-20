from flask import Flask
from PIL import Image, ImageFilter
import numpy as np
from keras.models import load_model
from keras.optimizers import SGD
from flask import Flask, jsonify, request
import io

def blur_image(pil_im):
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=3))
    blur_img = blur_img.resize((80, 80))
    return blur_img


app = Flask(__name__)

model = load_model('top_model.h5', compile=False)
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

fonts = ['Badscript', 'Amiriquran', 'Angkor', 'Antic', 'Amita', 'Anonymouspro', 'Anaheim', 'Amiri', 'Annieuseyourtelescope', 'Andika', 'Anticslab', 'Anticdidone', 'Architectsdaughter', 'Archivoblack', 'Arbutusslab', 'Arefruqaa', 'Anton', 'Arbutus', 'Aoboshione', 'Arapey', 'Arya', 'Arefruqaaink', 'Armata', 'Asar', 'Areyouserious', 'Asapcondensed', 'Arizonia', 'Arvo', 'Artifika', 'Arsenal', 'Autourone', 'Atkinsonhyperlegible', 'Audiowide', 'Asset', 'Atomicage', 'Astloch', 'Aubrey', 'Asul', 'Atma', 'Athiti', 'Averiasanslibre', 'Bacasimeantique', 'Averiaseriflibre', 'B612', 'Averagesans', 'B612mono', 'Babylonica', 'Averiagruesalibre', 'Average', 'Averialibre']

def results_to_label(probabilities):
    top_indices = np.argsort(probabilities)[-10:][::-1]
    top_results = [(float(probabilities[i]), fonts[i]) for i in top_indices]
    return top_results

@app.route('/process_list', methods=['POST'])
def process_list():
    # Check if the 'image' key is in the request.files dictionary
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' key found in the request."}), 400

    image_file = request.files['image']

    # Check if the file is not empty
    if image_file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Read the image file and process it
        pil_im = Image.open(io.BytesIO(image_file.read())).convert('L')  # Convert to grayscale
        pil_im = blur_image(pil_im)

        org_img = np.array(pil_im)
        data = org_img.reshape((1,) + org_img.shape) / 255.0  # Example reshaping and normalization

        # Assuming 'model' and 'results_to_label' functions are defined elsewhere
        y = model.predict(data)
        label = results_to_label(y[0])

        # You might want to convert the processed image to a base64-encoded string or another format
        # before returning it in the response, depending on your requirements.

        return jsonify({"result": label, "message": "Image processed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
