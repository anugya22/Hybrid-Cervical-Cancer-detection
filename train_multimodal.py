# train_multimodal.py
# I wired your image augmentation, tabular preprocessing, and fusion training loop into a Keras Sequence-based trainer.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from data_load import load_and_prepare
from models import build_image_model, TabularNet

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MultiModalSequence(Sequence):
    # I implemented a sequence that yields ([images, tabular], labels) using dataframe rows and a global tabular array.
    def __init__(self, df, tabular_array, image_datagen, batch_size=16, target_size=(224,224), shuffle=True):
        self.df = df.reset_index(drop=False).copy()  # keep original index in 'index' column
        self.tabular = tabular_array
        self.image_datagen = image_datagen
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.df))
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]
        images = []
        tabs = []
        labels = []
        for _, row in batch_df.iterrows():
            img_path = row['image_path']
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img = self.image_datagen.random_transform(img)
            images.append(img)
            original_index = int(row['index'])  # original df index
            tabs.append(self.tabular[original_index])
            labels.append(row['label'])
        images = np.stack(images, axis=0).astype('float32')
        tabs = np.stack(tabs, axis=0).astype('float32')
        labels = np.array(labels).astype('int32')
        return [images, tabs], labels

def build_fusion_model(tab_feature_dim):
    # Re-create the same architecture you used in notebook (ResNet50 projection + TabularNet + fusion)
    image_model = build_image_model(input_shape=(224,224,3), projection_dim=512)
    tab_net = TabularNet(input_dim=tab_feature_dim, hidden_dims=(128,64), out_dim=64)
    img_input = image_model.input
    tab_input = tf.keras.Input(shape=(tab_feature_dim,), name='tab_input')
    img_emb = image_model(img_input)
    tab_emb = tab_net(tab_input)
    x = layers.Concatenate()([img_emb, tab_emb])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=[img_input, tab_input], outputs=out)
    return model

def main(csv_path='data/labels.csv', out_dir='outputs', batch_size=16, epochs=20):
    os.makedirs(out_dir, exist_ok=True)
    df, tabular_array, train_df, val_df, tabular_cols = load_and_prepare(csv_path,
                                                                          image_col='image_path',
                                                                          label_col='label')
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    # Build model
    model = build_fusion_model(tab_feature_dim=tabular_array.shape[1])
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Create sequences
    train_seq = MultiModalSequence(train_df, tabular_array, image_datagen, batch_size=batch_size, target_size=(224,224))
    val_seq = MultiModalSequence(val_df, tabular_array, image_datagen, batch_size=batch_size, target_size=(224,224), shuffle=False)
    # Callbacks
    checkpoint = ModelCheckpoint(os.path.join(out_dir, 'hybrid_cervical_model.h5'), save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')
    model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=[checkpoint, es])
    # Save final model
    model.save(os.path.join(out_dir, 'hybrid_cervical_model_final.h5'))
    return model

if __name__ == "__main__":
    main()
