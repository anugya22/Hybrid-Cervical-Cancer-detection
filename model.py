# models.py
# I implemented the image backbone (ResNet50 projection) and the tabular MLP exactly as in your notebook.
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_image_model(input_shape=(224,224,3), projection_dim=512):
    base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    base.trainable = False
    inputs = base.input
    pooled = base.output
    proj = layers.Dense(projection_dim, activation='relu')(pooled)
    proj = layers.Dropout(0.3)(proj)
    image_model = models.Model(inputs=inputs, outputs=proj, name='image_backbone_proj')
    return image_model

class TabularNet(models.Model):
    # I used the same MLP architecture and dropout/batchnorm settings from your notebook.
    def __init__(self, input_dim, hidden_dims=(128,64), out_dim=64):
        super(TabularNet, self).__init__()
        self.dense1 = layers.Dense(hidden_dims[0])
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.drop1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(hidden_dims[1])
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.drop2 = layers.Dropout(0.2)
        self.out = layers.Dense(out_dim)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        return self.out(x)
