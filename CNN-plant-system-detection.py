r"""
Problem Statement:
Design a plant disease detection system using a Convolutional Neural Network (CNN). 
Uses a subset of the PlantVillage dataset (Tomato diseases).

Line-by-Line Execution Mapping for Examiner (Assignment 7: CNN on PlantVillage)

Data Ingestion & Pre-Processing
- Lines 21-23: Checks for the existence of the PlantVillage repository. If absent, executes a system-level git clone command to download the raw dataset from GitHub.
- Lines 25-45: Defines the 4 specific target classes. The loop mathematically isolates exactly 800 images per class and copies them into a new working directory. This enforces strict class balancing and bounds the computational load for the local CPU.
- Line 48: splitfolders.ratio() parses the isolated directory and strictly segregates the image files into 70% training data, 15% validation data, and 15% testing data.

Data Augmentation & Generators
- Lines 51-52: Defines static dimension hyperparameters. Resizes all input tensors to 128x128 pixels and limits mini-batch processing to 32 images per cycle.
- Lines 54-59: ImageDataGenerator defines the stochastic affine transformation matrix for training. It applies pixel normalization (1.0/255), 20-degree rotations, horizontal flipping, and 20% zooms to mathematically alter the pixel grid and prevent exact spatial memorization.
- Line 60: Initializes a separate generator for validation and testing that strictly executes mathematical pixel normalization (1.0/255) without applying any geometric transformations.
- Lines 62-64: flow_from_directory() dynamically loads the images from the hard drive into RAM in batches of 32 during the training loop. It automatically extracts and maps the class labels based on the subfolder nomenclature.

Model Architecture (Forward Pass Definition)
- Line 69: Sequential() initializes the linear stack framework for the neural network.
- Lines 70-72: Conv2D applies 32 separate 3x3 kernels to extract spatial features. BatchNormalization() actively standardizes the output tensor of this layer to have a mean of 0 and a variance of 1, preventing internal covariate shift. MaxPooling2D downsamples the spatial dimensions by a factor of 2.
- Lines 74-79: Executes two subsequent convolutional blocks, increasing the feature map depth to 64, then 128 kernels, each followed by batch normalization and max pooling to further compress the spatial resolution.
- Lines 81-84: Flatten() reshapes the 3D tensor into a 1D vector. Dense(256) applies non-linear feature combination. Dropout(0.5) applies a Bernoulli distribution mask to randomly zero out exactly 50% of the node connections to structurally prevent overfitting. Dense(num_classes) utilizes the softmax activation function to output a strictly bounded probability distribution summing to 1.0 across the 4 classes.

Compilation & Training
- Lines 87-91: compile() sets the parameters for the backward pass calculation. loss="categorical_crossentropy" calculates the error vector. The Adam optimizer is utilized with an explicitly reduced learning rate step size (0.0001) to prevent the gradients from overshooting the optimal minimum.
- Line 93: EarlyStopping() monitors the val_loss metric at the end of each epoch. The 'patience=5' parameter instructs the loop to automatically terminate if the validation loss fails to mathematically decrease for 5 consecutive epochs. 'restore_best_weights=True' reverts the network to its optimal state.
- Line 96: fit() initiates the training execution, streaming augmented data from train_gen for a maximum of 20 epochs.

Evaluation & Visualization
- Lines 99-108: Extracts the scalar history metrics from the completed training loop. matplotlib renders 2D line plots comparing training accuracy against validation accuracy, and training loss against validation loss, to visually verify convergence.
- Lines 110-111: evaluate() executes a strict forward pass on the unseen test_gen dataset and outputs the final objective accuracy metric.
- Lines 113-114: predict() outputs the raw probability tensors for the test set. np.argmax(axis=1) extracts the index position of the highest scalar probability to convert the output into a discrete integer prediction.
- Lines 115-121: confusion_matrix() calculates the mathematical cross-tabulation of the actual test labels against the predicted test labels. sns.heatmap renders the matrix into a red-scaled GUI window for visual classification verification.
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import splitfolders
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

# STEP 1: DATA INGESTION
if not os.path.exists('PlantVillage-Dataset'):
    print("Cloning dataset from GitHub... (This may take a few minutes depending on bandwidth)")
    os.system("git clone https://github.com/spMohanty/PlantVillage-Dataset.git")

src_root = 'PlantVillage-Dataset/raw/color'
dst_root = 'tomato_disease_subset'

if os.path.exists(dst_root):
    shutil.rmtree(dst_root)
os.makedirs(dst_root, exist_ok=True)

selected_classes = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

IMAGES_PER_CLASS = 800

for cls in selected_classes:
    src_dir = os.path.join(src_root, cls)
    dst_dir = os.path.join(dst_root, cls)
    os.makedirs(dst_dir, exist_ok=True)
    all_images = os.listdir(src_dir)
    random.seed(42)
    chosen = random.sample(all_images, min(IMAGES_PER_CLASS, len(all_images)))
    for img in chosen:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    print(f"Copied {len(chosen)} images for: {cls}")

# STEP 2: SPLIT DATA
splitfolders.ratio(dst_root, output='tomato_split', seed=42, ratio=(0.7, 0.15, 0.15))

# STEP 3: GENERATORS & AUGMENTATION
IMG_SIZE = 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory('tomato_split/train', target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
val_gen = val_test_datagen.flow_from_directory('tomato_split/val', target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
test_gen = val_test_datagen.flow_from_directory('tomato_split/test', target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

# STEP 4: ARCHITECTURE
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# STEP 5: COMPILE & TRAIN
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nStarting training...")
history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[early_stop])

# STEP 6: EVALUATION & VISUALIZATION
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss')
axes[1].legend()
plt.tight_layout()
plt.show()

test_loss, test_acc = model.evaluate(test_gen)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

y_pred = np.argmax(model.predict(test_gen), axis=1)
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Tomato Disease')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

"""
-----------------------------------------------------------------------------
1. Visual Plot & Terminal Output Analysis
-----------------------------------------------------------------------------
A. The Accuracy Plot (Left Subplot)
* Train Accuracy (Blue Line): Demonstrates rapid logarithmic convergence 
  approaching 1.0 (100%). The network successfully maps the features of the 
  training dataset almost immediately.
* Val Accuracy (Orange Line): Remains completely static at exactly 0.25 for 
  the first 4 epochs. Because this dataset contains exactly 4 classes, an 
  accuracy of 0.25 indicates a random probability distribution. The network 
  fails to extract generalizable spatial features initially. Between epochs 
  4 and 8, it experiences rapid convergence, peaking near 0.92 before slightly 
  degrading.

B. The Loss Plot (Right Subplot)
* Train Loss (Blue Line): Exhibits an exponential decay curve, quickly 
  flatlining near 0.0. This confirms the Adam optimizer has successfully 
  minimized the Categorical Crossentropy error on the training tensors.
* Val Loss (Orange Line): Exhibits extreme initial instability, spiking to 
  nearly 10.0 around epoch 2, indicating massive gradient updates that failed 
  to generalize. It then descends rapidly, reaching its absolute mathematical 
  minimum at Epoch 9 (val_loss: 0.2042).

C. The Overfitting Divergence & Early Stopping
* The Divergence: The defining characteristic of the Loss plot occurs exactly 
  after Epoch 9. The blue Train loss remains flat near zero, but the orange 
  Val loss strictly and continuously increases (terminating at 0.6926 at 
  Epoch 14). This specific mathematical divergence is the definitive visual 
  proof of overfitting: the weight matrices are actively updating to memorize 
  training data at the direct expense of validation generalization.
* Callback Execution: The EarlyStopping algorithm continuously monitored the 
  Val loss line. Triggered by the 'patience=5' parameter, it tracked exactly 
  5 consecutive epochs of upward trajectory (Epochs 10, 11, 12, 13, and 14). 
  At Epoch 14, it mathematically proved no further global minimums would be 
  reached, terminated the training loop, and restored the optimal network 
  state from Epoch 9.

-----------------------------------------------------------------------------
2. Deep Learning Concepts: Advanced CNN Architecture
-----------------------------------------------------------------------------
A. Data Augmentation (ImageDataGenerator)
* Mechanics: A regularization technique that dynamically applies stochastic 
  affine transformations (e.g., 20-degree rotation, horizontal flipping, 
  20% zoom) to the input tensors during the training loop.
* Objective: Prevents the network from memorizing exact pixel coordinates, 
  forcing the convolutional filters to learn invariant spatial features and 
  drastically reducing variance.

B. Batch Normalization
* Mechanics: Standardizes the mathematical activations of the previous layer 
  for each mini-batch. It subtracts the batch mean and divides by the standard 
  deviation.
  Formula: x_norm = (x - batch_mean) / sqrt(batch_variance + epsilon)
* Objective: Mitigates internal covariate shift, smoothing the overall loss 
  landscape and allowing for accelerated convergence during gradient descent.

C. Early Stopping Regularization
* Mechanics: A dynamic algorithm that actively monitors validation metrics 
  (like val_loss) to halt the training loop prior to executing all epochs.
* Objective: Acts as an automated mathematical safeguard against overfitting.

D. Optimizer: Adam (Adaptive Moment Estimation)
* Mechanics: Computes individual adaptive learning rates based on the first 
  moment (exponential moving average of the gradient) and the second moment 
  (squared gradient).
* Reduced Learning Rate (0.0001): A specifically constrained step size (10^-4) 
  forces micro-adjustments to the weights, preventing the gradient vectors 
  from wildly overshooting the global minimum.
"""