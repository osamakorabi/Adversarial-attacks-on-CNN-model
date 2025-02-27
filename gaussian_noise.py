import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- Data Preparation ---

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train / 255.0  # Normalize to [0, 1]
x_test = x_test / 255.0

# Add a channel dimension (needed for CNN input)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# --- CNN Model Definition ---

def create_cnn_model():
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flattening layer
        tf.keras.layers.Flatten(),

        # Fully connected dense layer
        tf.keras.layers.Dense(units=128, activation='relu'),

        # Output layer with 10 neurons (one for each digit class)
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    return model

# Create the model
model = create_cnn_model()

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print the model summary
model.summary()

# --- Training the Model ---

# Train the model
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model for future use
model.save("mnist_cnn_model.h5")

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- Gaussian Noise Attack ---

# Function to add Gaussian noise
def add_gaussian_noise(x, mean=0.0, stddev=0.1):
    """
    Add Gaussian noise to the input images.
    :param x: Input images
    :param mean: Mean of the Gaussian noise
    :param stddev: Standard deviation of the Gaussian noise
    :return: Noisy images
    """
    noise = np.random.normal(mean, stddev, x.shape)
    noisy_images = x + noise
    return np.clip(noisy_images, 0, 1)  # Ensure [0, 1] range

# Generate noisy images
mean = 0.0
stddev = 0.5  # Standard deviation controls noise strength

x_sample = x_test[:5]  # Take first 5 test images
y_sample = y_test[:5]

noisy_examples = add_gaussian_noise(x_sample, mean, stddev)

# Evaluate the model on noisy examples
noisy_loss, noisy_accuracy = model.evaluate(noisy_examples, y_sample)
print(f"Noisy Loss: {noisy_loss}")
print(f"Noisy Accuracy: {noisy_accuracy}")

# --- Visualization: Original vs Noisy Images ---
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(noisy_examples[i].reshape(28, 28), cmap='gray')
    plt.title("Gaussian Noise")
    plt.axis('off')

plt.tight_layout()
plt.show()
