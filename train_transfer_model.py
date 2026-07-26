"""
Fruit Quality Detection - Transfer Learning Training Script
Uses a pre-trained MobileNetV2 base (ImageNet weights) with a custom
classification head, per the report's use of transfer learning
(Chapters 1, 4, and Literature Review's mention of pre-trained CNNs).
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"

# ----------------------------------------------------------------------
# Data loading + augmentation
# ----------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

num_classes = train_generator.num_classes
print("Detected classes:", train_generator.class_indices)

# ----------------------------------------------------------------------
# Transfer learning: MobileNetV2 base (frozen) + custom head
# ----------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze pre-trained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------------------------------------------------
# Train (only the new head layers, base is frozen)
# ----------------------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
)

# ----------------------------------------------------------------------
# Save trained model
# ----------------------------------------------------------------------
model.save("fruit_classifier_model.h5")
print("Model saved as fruit_classifier_model.h5")
