import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Load the MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()

# Normalize the input data
x_train = x_train.astype("float32") / 255.0

# Reshape the input data to add channel dimension for grayscale images
x_train = x_train.reshape((-1, 28, 28, 1))

# Define the generator model
generator = keras.Sequential(
    [
        keras.Input(shape=(100,)),
        layers.Dense(7 * 7 * 128),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

# Define the discriminator model
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1),
    ],
    name="discriminator",
)
discriminator.summary()

# Combine the generator and discriminator into a GAN
gan = keras.Sequential([generator, discriminator])

# Compile the discriminator (as a standalone model)
discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Compile the GAN
gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Training loop
batch_size = 32
epochs = 10
steps_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step in range(steps_per_epoch):
        # Generate random noise as input to the generator
        noise = tf.random.normal(shape=(batch_size, 100))

        # Generate images using the generator
        generated_images = generator(noise)

        # Create a batch by sampling real images from the training set
        real_images = x_train[np.random.choice(x_train.shape[0], size=batch_size, replace=False)]

        # Concatenate real and generated images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Labels for generated and real images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels (important trick)
        labels += 0.05 * tf.random.uniform(labels.shape)

        # Train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # Generate noise as input to the GAN
        noise = tf.random.normal(shape=(batch_size, 100))

        # Labels for generated images (trick the discriminator)
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the GAN (generator only)
        g_loss = gan.train_on_batch(noise, misleading_labels)

    # Print the losses
    print(f"Discriminator loss: {d_loss:.4f}")
    print(f"Generator loss: {g_loss:.4f}")
    print()

# Generate some images using the trained generator
num_samples = 10
noise = tf.random.normal(shape=(num_samples, 100))
generated_images = generator.predict(noise)

# Display the generated images
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, num_samples, figsize=(20, 2))
fig.suptitle("Generated Images")

for i in range(num_samples):
    axs[i].imshow(generated_images[i].reshape(28, 28), cmap="gray")
    axs[i].axis("off")

plt.show()
