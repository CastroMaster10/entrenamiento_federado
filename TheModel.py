import tensorflow as tf

class build:
    @staticmethod
    def build_it():
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))

        # simple normalisation
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

        # Conv Block 1
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        # Conv Block 2
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        # Head
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model