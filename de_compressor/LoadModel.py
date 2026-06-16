import os
from typing import Any

# SHUT UPP TENSORFLOWWWW, IM THROUGH WITH YOU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Make the AI deterministic
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import tensorflow as tf

from NextLetterModel import NextLetterModel
from OneStep import OneStep

vocabulary = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]
idsFromChars = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), mask_token=None)
charsFromIds = tf.keras.layers.StringLookup(vocabulary=idsFromChars.get_vocabulary(), invert=True, mask_token=None)

vocabSize = len(idsFromChars.get_vocabulary())
embeddingDim = 256
rnnUnits = 1024

def loadModel():
    model = NextLetterModel(vocabSize, embeddingDim, rnnUnits)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    dummyData = tf.zeros((1,1), dtype=tf.int32)
    model(dummyData) # dummy data to build weights, needed before loading weights
    model.load_weights("./llm/model.weights.h5")
    return model

def getOneStepGenerator(model):
    return OneStep(model, charsFromIds, idsFromChars)

def generateNextCharacter(oneStepReloaded, currentString: str, states=None) -> tuple[str, Any]:
    nextChar = tf.constant([ currentString ])
    nextChars, states = oneStepReloaded.generateTopKNextCharacters(nextChar, states=states, k=12)
    predictedCharacter = nextChars[0].numpy()[0].decode("utf-8")
    
    nextChars = [ a.numpy()[0].decode("utf-8") for a in nextChars ]
    
    """
    nextChar = tf.constant([ currentString ])
    nextChar, states = oneStepReloaded.generate_one_step(nextChar, states=states)
    predictedCharacter = nextChar.numpy()[0].decode("utf-8")
    """
    return (predictedCharacter, states, nextChars)

#model = loadModel()
#oneStepReloaded = getOneStepGenerator(model)
