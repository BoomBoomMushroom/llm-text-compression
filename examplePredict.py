from typing import Any

import os
import numpy as np
import random
import time

# SHUT UPP TENSORFLOWWWW, IM THROUGH WITH YOU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Make the AI deterministic
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import tensorflow as tf

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

# RLE time?

class BitReader:
    def __init__(self, data: bytearray):
        self.data = data
        self.bitPos = 0
    
    def readBit(self) -> int:
        byteIndex = self.bitPos // 8
        bitIndex = 7 - (self.bitPos % 8) # so we read MSB first
        
        byte = self.data[byteIndex]
        bit = (byte >> bitIndex) & 1
        
        self.bitPos += 1
        return bit

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

def readVarInt4(data: bytearray) -> tuple[int, int]:
    SEGMENT_BITS = 0x07
    CONTINUE_BIT = 0x08
    value = 0
    shift = 0
    i = 0
    
    while True:
        b = data[i]
        i += 1
        value |= (b & SEGMENT_BITS) << shift
        
        if (b & CONTINUE_BIT) == 0: break
        shift += 3
    
    return value, i

def writeVarInt4(value: int) -> bytearray:
    SEGMENT_BITS = 0x07
    CONTINUE_BIT = 0x08
    out = bytearray()
    
    while True:
        if (value & ~SEGMENT_BITS) == 0:
            out.append(value)
            break
        
        out.append((value & SEGMENT_BITS) | CONTINUE_BIT)
        value >>= 3
    
    return out


"""
binString = "10111100000001000000000100010000110000000000000101010000001011100000010000011000000000011110011100101001010101"
binStringLength = len(binString)
char = "1"
digitCounter = 0
data = []
while len(binString) > 0:
    digit = binString[0]
    if digit == char:
        digitCounter += 1
    else:
        data.append( (char, digitCounter) )
        char = digit
        digitCounter = 1
    
    binString = binString[1:]

print(data)

rleVarIntBits = 0
nybbles = 0

# todo: take each one and pack these varint4's together and then we should have our ultimately compressed data <3
for _, c in data:
    byte: bytearray = writeVarInt(c)
    print(f"{c} -> ", end="")
    for b in byte:
        nybbles += 1
        print(f"{hex(b)} ", end="")
    print()

print(f"nybbles: {nybbles} | bytes: {nybbles/2} | original length: {binStringLength/8} Bytes")
exit()
"""

for c in vocabulary:
    char = bin(c.encode("utf-8")[0])
    char = "0b" + char.split("0b")[1].zfill(8)
    print(f"{c} -> {char}")

exit()


def generateNextCharacter(currentString: str, states=None) -> tuple[str, Any]:
    nextChar = tf.constant([ currentString ])
    nextChars, states = oneStepReloaded.generateTopKNextCharacters(nextChar, states=states, k=4)
    predictedCharacter = nextChars[0].numpy()[0].decode("utf-8")
    
    nextChars = [ a.numpy()[0].decode("utf-8") for a in nextChars ]
    
    """
    nextChar = tf.constant([ currentString ])
    nextChar, states = oneStepReloaded.generate_one_step(nextChar, states=states)
    predictedCharacter = nextChar.numpy()[0].decode("utf-8")
    """
    return (predictedCharacter, states, nextChars)

def generateCompressedText(stringToEncode: str) -> list[str]:
    states = None
    out = [stringToEncode[0]]

    # 1st character is already in
    for n in range(1, len(stringToEncode)):
        #predictedCharacter, states = generateNextCharacter(stringToEncode[n-1], states)
        predictedCharacter, states, topPredictedCharacters = generateNextCharacter(stringToEncode[n-1], states)
        correctCharacter = stringToEncode[n]
        
        if correctCharacter in topPredictedCharacters:
            index = topPredictedCharacters.index(correctCharacter)
            out.append(index)
        else:
            # Wrong letter, lets correct it
            predictedCharacter = correctCharacter # why is this here? idk but ill leave it just in case
            out.append(correctCharacter)
    
    return out

def compressText(text: str, writeDashedText=False) -> bytearray:
    letters: list[str] = generateCompressedText(text)
    
    stringBitMask = 0b0
    
    dataBytes: int = 0
    dataShifted = 0
    for l in letters:
        wasAiCorrect = type(l)==int
        
        stringBitMask <<= 1
        # write a 0 if we need to predict it, or a 1 if we explicitly need to write the character
        bit = 0 if wasAiCorrect else 1
        stringBitMask |= bit
        
        if wasAiCorrect:
            # huffman tree for top 4 results
            dataBytes <<= (l+1)
            dataShifted += (l+1)
            
            if l == 0:
                #dataBytes <<= 1
                #dataShifted += 1
                dataBytes |= 0b0
            if l == 1:
                #dataBytes <<= 2
                #dataShifted += 2
                dataBytes |= 0b10
            if l == 2:
                #dataBytes <<= 3
                #dataShifted += 3
                dataBytes |= 0b110
            if l == 3:
                #dataBytes <<= 4
                #dataShifted += 4
                dataBytes |= 0b1110
        else:
            dataBytes <<= 8
            dataShifted += 8
            dataBytes |= l.encode("utf-8")[0]
    
    dataBytes <<= (8 - (dataShifted % 8)) # align the data to a byte
    
    dataByteArray: bytearray = bytearray()
    while dataShifted > 0:
        b = dataBytes & 0xFF
        dataByteArray += bytearray(int.to_bytes(b))
        
        dataBytes >>= 8
        dataShifted -= 8
    dataByteArray = dataByteArray[::-1] # reverse it since we put it in backwards
    
    stringBitMaskAsBytes: bytearray = writeVarInt(stringBitMask)
    out: bytearray = stringBitMaskAsBytes + dataByteArray
    
    if writeDashedText:
        dashedString = ['-' if type(c) == int else c for c in letters]
        dashedString = ''.join(dashedString)
        
        print(dashedString)
        print(bin(stringBitMask).split("b")[1])
    
    return out

def decompressText(data: bytearray) -> str:
    stringBitMask, bytesRead = readVarInt(data)
    print(f"Bytes for mask: {bytesRead}")
    data = data[bytesRead:]
    
    bitReader = BitReader(data)
    
    bitMaskStr = bin(stringBitMask).split("b")[1]
    out = ""
    llmStates = None
    for bit in bitMaskStr:
        nextChar = None
        
        # use the AI anyways, even if we're not going to use it's output since we need the state tree to be the same
        #if len(out) > 0: nextChar, llmStates = generateNextCharacter(out[-1], llmStates)
        if len(out) > 0: nextChar, llmStates, topPredictedCharacters = generateNextCharacter(out[-1], llmStates)
        
        if bit == "1":
            byte = 0b0
            for i in range(0, 8):
                byte <<= 1
                byte |= bitReader.readBit()
            
            nextChar = bytes([byte]).decode("utf-8")
        else:
            index = 0
            while True:
                bit = bitReader.readBit()
                if bit == 0: break
                index += 1
            nextChar = topPredictedCharacters[index]
        
        out += nextChar
        
        """
        if bit == "1":
            nextChar = bytes([data[0]]).decode("utf-8")
            data = data[1:]
        out += nextChar
        """
    
    print()
    
    return out

# idea: use RLE to compress that monster of a number out front. rle should work well since its a bit mask and has long runs of 0s or 1s
compressed = compressText(stringToEncode, True)
print("Compressed: ", compressed, "\n")

with open("out.txt", "wb") as f: f.write(compressed)

decompressed = decompressText(compressed)

print(f"Results equal? {stringToEncode==decompressed}")
print(decompressed)

compressedSize = len(compressed)
decompressedSize = len(bytearray(decompressed.encode("utf-8")))

print(compressedSize, decompressedSize, (1-(compressedSize / decompressedSize)), compressedSize < decompressedSize)
#"""
