import tensorflow as tf

class NextLetterModel(tf.keras.Model):
    def __init__(self, vocabSize, embeddingDim, rnnUnits):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocabSize, embeddingDim)
        self.gru = tf.keras.layers.GRU(rnnUnits, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocabSize)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states == None:
            #states = self.gru.get_initial_state(x)
            states = self.gru.get_initial_state(batch_size=tf.shape(x)[0])
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        
        if return_state: return x, states
        else: return x