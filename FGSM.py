import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# Define the CNN model
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

# --- FGSM Adversarial Attack ---
# Function to create adversarial examples using FGSM
def create_adversarial_example(model, x, y, epsilon=0.1):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)
    
    # Get gradients of loss w.r.t input
    gradient = tape.gradient(loss, x_tensor)
    
    # Get the sign of the gradients
    signed_grad = tf.sign(gradient)
    
    # Create adversarial example
    adversarial_example = x + epsilon * signed_grad
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)  # Ensure values are valid [0,1]
    return adversarial_example.numpy()

# Generate adversarial examples for some test samples
epsilon = 0.1  # Strength of the adversarial perturbation
x_sample = x_test[:5]  # Select a few test samples
y_sample = y_test[:5]

adversarial_examples = create_adversarial_example(model, x_sample, y_sample)

# Evaluate the model on the adversarial examples
adversarial_loss, adversarial_accuracy = model.evaluate(adversarial_examples, y_sample)
print(f"Adversarial Loss: {adversarial_loss}")
print(f"Adversarial Accuracy: {adversarial_accuracy}")

# --- Visualizing Original and Adversarial Examples ---
# Plot original and adversarial examples
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(adversarial_examples[i].reshape(28, 28), cmap='gray')
    plt.title("Adversarial")
    plt.axis('off')

plt.tight_layout()
plt.show()
