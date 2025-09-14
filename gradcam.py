# gradcam.py
# I implemented a Keras-compatible Grad-CAM helper compatible with EfficientNetB0.
# For EfficientNetB0, try last_conv_layer_name="block7a_project_conv" (common final conv-layer).

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: (1, H, W, C) preprocessed as during training (0..1)
    model: Keras model accepting [image, tabular] or single-image input
    last_conv_layer_name: string name of conv layer in backbone (e.g., 'block7a_project_conv')
    """
    image_input = model.inputs[0]
    grad_model = Model([image_input], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def apply_heatmap_on_image(img_path, heatmap, alpha=0.4, target_size=(224,224)):
    img = cv2.imread(img_path)[:,:,::-1]
    img = cv2.resize(img, target_size)
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap = np.uint8(255 * hmap)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    superimposed = hmap_color * alpha + img
    superimposed = np.clip(superimposed / superimposed.max(), 0, 1)
    return np.uint8(255 * superimposed)
