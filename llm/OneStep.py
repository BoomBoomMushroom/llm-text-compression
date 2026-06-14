# code from https://www.tensorflow.org/text/tutorials/text_generation
import tensorflow as tf

class OneStep(tf.keras.Model):
    def __init__(self, model, charsFromIds, idsFromChars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.charsFromIds = charsFromIds
        self.idsFromChars = idsFromChars

        # Create a mask to prevent "[UNK]" from being generated.
        skipIds = self.idsFromChars(['[UNK]'])[:, None]
        sparseMask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skipIds),
            indices=skipIds,
            # Match the shape to the vocabulary
            dense_shape=[len(idsFromChars.get_vocabulary())])
        self.predictionMask = tf.sparse.to_dense(sparseMask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        inputChars = tf.strings.unicode_split(inputs, 'UTF-8')
        inputIds = self.idsFromChars(inputChars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predictedLogits, states = self.model(inputs=inputIds, states=states, return_state=True)
        # Only use the last prediction.
        predictedLogits = predictedLogits[:, -1, :]
        predictedLogits = predictedLogits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predictedLogits = predictedLogits + self.predictionMask

        # Sample the output logits to generate token IDs.
        predictedIds = tf.random.categorical(predictedLogits, num_samples=1)
        predictedIds = tf.squeeze(predictedIds, axis=-1)

        # Convert from token ids to characters
        predictedChars = self.charsFromIds(predictedIds)

        # Return the characters and model state.
        return predictedChars, states
