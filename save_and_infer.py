# save_and_infer.py
# I provide a small inference helper that loads the saved hybrid model and predicts for a single sample.

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_image(path, target=(224,224)):
    img = Image.open(path).convert('RGB').resize(target)
    arr = np.array(img).astype('float32') / 255.0
    return arr

def predict_single(model_path, image_path, tab_vector):
    model = load_model(model_path)
    img = np.expand_dims(load_image(image_path), axis=0)
    tab = np.expand_dims(np.array(tab_vector, dtype='float32'), axis=0)
    preds = model.predict([img, tab])
    return preds

if __name__ == "__main__":
    # demo (replace with real file and tab_vector)
    model_path = 'outputs/hybrid_cervical_model.h5'
    image_path = 'data/example.jpg'
    tab_vector = [0.0] * 10
    print(predict_single(model_path, image_path, tab_vector))
