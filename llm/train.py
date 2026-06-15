import re
import os

vocabulary = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]

def cleanAndFilterText(text: str) -> str:
    allOneLine = " ".join(text.split("\n"))
    filteredContents = re.sub(r'[^a-zA-Z0-9 ]', '', allOneLine)
    filteredContents = " ".join(filteredContents.split()) # removes double, triple, or any amount of extra spaces in a row
    return filteredContents

# Load and clean up our data
trainingData = []
"""
with open("./data/alice_in_wonderland.txt", "r") as f:
    textContents = cleanAndFilterText( f.read() )
    trainingData.append(textContents)
"""

for (dirpath, dirnames, filenames) in os.walk("./data/wikipedia"):
    for file in filenames:
        with open(f"./data/wikipedia/{file}", "r") as f:
            textContents = cleanAndFilterText( f.read() )
            trainingData.append(textContents)


#print(filteredContents)

import tensorflow as tf
print("") # separate all the tf startup messages from mine
import numpy as np
import time

chars = tf.strings.unicode_split(" ".join(trainingData), input_encoding="UTF-8")
idsFromChars = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), mask_token=None)
charsFromIds = tf.keras.layers.StringLookup(vocabulary=idsFromChars.get_vocabulary(), invert=True, mask_token=None)

def textFromIds(ids):
    return tf.strings.reduce_join(charsFromIds(ids), axis=-1)

# Create training examples & targets

allIds = idsFromChars(chars)
idsDataset = tf.data.Dataset.from_tensor_slices(allIds)
seq_length = 100

sequences = idsDataset.batch(seq_length+1, drop_remainder=True)
"""
print("Sequences:")
for seq in sequences.take(5):
    print(textFromIds(seq).numpy())
"""

def splitInputTarget(sequence):
    inputText = sequence[:-1]
    targetText = sequence[1:]
    return inputText, targetText

dataset = sequences.map(splitInputTarget)
"""
for input_example, target_example in dataset.take(1):
    print("Input :", textFromIds(input_example).numpy())
    print("Target:", textFromIds(target_example).numpy())
"""

# Training batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000 # Buffer size to shuffle the dataset

dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

# Building the model
vocabSize = len(idsFromChars.get_vocabulary())
embeddingDim = 256
rnnUnits = 1024

from NextLetterModel import NextLetterModel

model = NextLetterModel(vocabSize=vocabSize, embeddingDim=embeddingDim, rnnUnits=rnnUnits)

"""
for inputExampleBatch, targetExampleBatch in dataset.take(1):
    exampleBatchPredictions = model(inputExampleBatch)
    print(exampleBatchPredictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
    sampledIndices = tf.random.categorical(exampleBatchPredictions[0], num_samples=1)
    sampledIndices = tf.squeeze(sampledIndices, axis=-1).numpy()
    
    print(sampledIndices)
    print("Input:\n", textFromIds(inputExampleBatch[0]).numpy())
    print()
    print("Next Char Predictions:\n", textFromIds(sampledIndices).numpy())

print(model.summary())
"""


# Configure training

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

"""
for inputExampleBatch, targetExampleBatch in dataset.take(1):
    exampleBatchPredictions = model(inputExampleBatch)
    
    exampleBatchMeanLoss = loss(targetExampleBatch, exampleBatchPredictions)
    print("Prediction shape: ", exampleBatchPredictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", exampleBatchMeanLoss)
    
    print(tf.exp(exampleBatchMeanLoss).numpy())
"""

model.compile(optimizer='adam', loss=loss)

checkpointDir = './llm/trainingCheckpoints'
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPrefix, save_weights_only=True)

# Train the model
EPOCHS = 150
loadModel = None
#loadModel = "./llm/trainingCheckpoints/ckpt_10.weights.h5"
#loadModel = "./llm/model.weights.h5"

if loadModel == None or loadModel == "":
    print("Starting training...")
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
    print(f"Loading weights from `{loadModel}`")
    for input_example, _ in dataset.take(1): model(input_example) # force build weights
    model.load_weights(loadModel)

print("Saving weights to mode.weights.h5")
model.save_weights("./llm/model.weights.h5")

from OneStep import OneStep
oneStepModel = OneStep(model, charsFromIds, idsFromChars)

#"""
# Generate text
start = time.time()
states = None
nextChar = tf.constant(['Programming is '])
result = [nextChar]

for n in range(1000):
    nextChar, states = oneStepModel.generate_one_step(nextChar, states=states)
    result.append(nextChar)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
#"""



"""
# Load model and generate text
model = NextLetterModel(vocabSize, embeddingDim, rnnUnits)
model.build(tf.TensorShape([None, None]))
model.load_weights("./llm/model.weights.h5")
oneStepReloaded = OneStep(model, charsFromIds, idsFromChars)

states = None
next_char = tf.constant(['Alice '])
result = [next_char]

for n in range(100):
    next_char, states = oneStepReloaded.generate_one_step(next_char, states=states)
    result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))
"""
