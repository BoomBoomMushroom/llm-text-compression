from typing import Any

import os
import numpy as np
import random
import time

import tensorflow as tf

# SHUT UPP TENSORFLOWWWW, IM THROUGH WITH YOU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Make the AI deterministic
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

def setModelSeed(newSeed: int):
    random.seed(newSeed)
    np.random.seed(newSeed)
    tf.random.set_seed(newSeed)

DEFAULT_SEED = 0
setModelSeed(DEFAULT_SEED)

stringToEncode = "coda started making their obelisks and then theo made a lavacast and now those are the \"Rockets\" space X makes"
#stringToEncode = input("Enter string to encode: ")

# Load Model
from llm.NextLetterModel import NextLetterModel
from llm.OneStep import OneStep

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

model = loadModel()
oneStepReloaded = getOneStepGenerator(model)

# Actually compress and decompress it and stuff

def readVarInt(dataBuffer: bytes) -> tuple[int, int]: # value, bytes read
    SEGMENT_BITS = 0x7F
    CONTINUE_BIT = 0x80
    value = 0
    position = 0
    currentByte = 0
    currentByteIndex = 0
    
    while True:
        currentByte = dataBuffer[currentByteIndex]
        currentByteIndex += 1
        
        value |= (currentByte & SEGMENT_BITS) << position
        if (currentByte & CONTINUE_BIT) == 0: break
        
        position += 7
        #if position >= 32: raise Exception("VarInt is too big!")
    
    return (value, currentByteIndex)

def writeVarInt(value: int) -> bytearray:
    SEGMENT_BITS = 0x7F
    CONTINUE_BIT = 0x80
    output = bytearray(0)
    
    while True:
        if (value & ~SEGMENT_BITS) == 0:
            output.append(value)
            break
        
        write = (value & SEGMENT_BITS) | CONTINUE_BIT
        output.append(write)
    
        value >>= 7
    
    return output

def generateNextCharacter(currentString: str, states=None) -> tuple[str, Any]:
    nextChar = tf.constant([ currentString ])
    nextChar, states = oneStepReloaded.generate_one_step(nextChar, states=states)
    predictedCharacter = nextChar.numpy()[0].decode("utf-8")
    return (predictedCharacter, states)

def generateCompressedText(stringToEncode: str) -> list[str]:
    states = None
    out = [stringToEncode[0]]

    # 1st character is already in
    for n in range(1, len(stringToEncode)):
        predictedCharacter, states = generateNextCharacter(stringToEncode[n-1], states)
        correctCharacter = stringToEncode[n]
        
        if predictedCharacter == correctCharacter:
            # Correctly predicted so we dont need to encode the character
            out.append(None)
        else:
            # Wrong letter, lets correct it
            predictedCharacter = correctCharacter
            out.append(correctCharacter)
    
    return out

def compressText(text: str, seed: int=0) -> bytearray:
    setModelSeed(seed)
    letters: list[str] = generateCompressedText(text)
    
    stringBitMask = 0b0
    lettersByteArray = bytearray(0)
    for l in letters:
        stringBitMask <<= 1
        # write a 0 if we need to predict it, or a 1 if we explicitly need to write the character
        bit = 0 if l == None else 1
        stringBitMask |= bit
        
        if l != None: lettersByteArray += bytearray(l.encode("utf-8"))
    
    stringBitMaskAsBytes: bytearray = writeVarInt(stringBitMask)
    
    out: bytearray = stringBitMaskAsBytes + lettersByteArray
    
    dashedString = ['-' if c == None else c for c in letters]
    dashedString = ''.join(dashedString)
    print(dashedString)
    print(bin(stringBitMask).split("b")[1])
    
    return out

def decompressText(data: bytearray, seed: int=0) -> str:
    setModelSeed(seed)
    stringBitMask, bytesRead = readVarInt(data)
    data = data[bytesRead:]
    
    bitMaskStr = bin(stringBitMask).split("b")[1]
    out = ""
    llmStates = None
    for bit in bitMaskStr:
        nextChar = None
        
        # use the AI anyways, even if we're not going to use it's output since we need the state tree to be the same
        if len(out) > 0: nextChar, llmStates = generateNextCharacter(out[-1], llmStates)
        
        if bit == "1":
            nextChar = bytes([data[0]]).decode("utf-8")
            data = data[1:]
        out += nextChar
    
    return out


useSeed = 10
compressed = compressText(stringToEncode, useSeed)
decompressed = decompressText(compressed, useSeed)

print(f"Results equal? {stringToEncode==decompressed}")
print(decompressed)
