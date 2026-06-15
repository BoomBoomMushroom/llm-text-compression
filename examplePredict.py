from typing import Any
import os

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

rankHistogram = {}
misses = 0
def printHistogram():
    totalHits = sum(rankHistogram.values())
    huffmanString = ""
    
    print("Histogram:")
    for rank in sorted(rankHistogram):
        count = rankHistogram[rank]
        percent = (count / totalHits) * 100
        print(f"Rank {rank}: {count:3d} ({percent:.1f}%)")
        
        huffmanString += vocabulary[rank] * count
    
    print("Huffman fit string:", huffmanString)
    print(f"Misses: {misses} | Hits: {totalHits}")

class BitReader:
    def __init__(self, data: bytearray):
        self.data = data
        self.bitPos = 0
    
    def countBitsRemaining(self) -> int:
        return (len(self.data)*8) - self.bitPos
    
    def readBit(self, advanceCounter: bool=True) -> int:
        byteIndex = self.bitPos // 8
        bitIndex = 7 - (self.bitPos % 8) # so we read MSB first
        
        byte = self.data[byteIndex]
        bit = (byte >> bitIndex) & 1
        
        if advanceCounter: self.bitPos += 1
        return bit

# Huffman coded "000000000000000000000000000000000000000000000011111111111122222222222222333334444455566777889  aaa"
# index: (binaryCode, bitsNeeded)
huffmanTree = {
    0: (0b0, 1),
    2: (0b110, 3),
    1: (0b101, 3),
    3: (0b1000, 4),
    4: (0b11111, 5),
    5: (0b10011, 5),
    7: (0b10010, 5),
    11: (0b11101, 5),
    6: (0b111101, 6),
    8: (0b111100, 6),
    10: (0b111001, 6),
    9: (0b111000, 6),
}

def encodeIndex(index: int) -> tuple[int, int]:
    dataOut = 0b0
    bitsWrote = 0
    
    #"""
    dataOut = huffmanTree[index][0]
    bitsWrote = huffmanTree[index][1]
    
    #print(dataOut, bitsWrote)
    #"""
    
    """
    # write the "read until we hit a 0" encoding
    bitsWrote = index+1
    if index == 0: dataOut = 0b0
    if index == 1: dataOut = 0b10
    if index == 2: dataOut = 0b110
    if index == 3: dataOut = 0b1110
    if index == 4: dataOut = 0b11110
    if index == 5: dataOut = 0b111110
    if index == 6: dataOut = 0b1111110 # worse case result, will give us an encoded size of 8 bits, or 1 byte ; which ties with encoding the entire letter
    #"""
    
    return (dataOut, bitsWrote)

def decodeIndex(bitReader: BitReader) -> int:
    index = 0
    
    #"""
    bitsRead = 0
    huffmanData = 0
    while bitsRead < 8:
        bitsRead += 1
        huffmanData <<= 1
        huffmanData |= bitReader.readBit()
        
        didSetIndex = False
        for key, value in huffmanTree.items():
            if bitsRead != value[1]: continue
            if huffmanData != value[0]: continue
            
            index = key
            didSetIndex = True
            break
        if didSetIndex: break
    #"""
    
    """
    # read until we hit a 0
    while True:
        bit = bitReader.readBit()
        if bit == 0: break
        index += 1
    #"""
    
    return index


def generateNextCharacter(currentString: str, states=None) -> tuple[str, Any]:
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

def generateCompressedText(stringToEncode: str) -> list[str]:
    global rankHistogram
    global misses
    
    states = None
    out = [stringToEncode[0]]

    # 1st character is already in
    for n in range(1, len(stringToEncode)):
        predictedCharacter, states, topPredictedCharacters = generateNextCharacter(stringToEncode[n-1], states)
        #predictedCharacter, states, topPredictedCharacters = generateNextCharacter(stringToEncode[:n], states)
        correctCharacter = stringToEncode[n]
        
        if correctCharacter in topPredictedCharacters:
            index = topPredictedCharacters.index(correctCharacter)
            out.append(index)
            
            rankHistogram[index] = rankHistogram.get(index, 0) + 1
        else:
            # Wrong letter, lets correct it
            out.append(correctCharacter)
            misses += 1
    
    return out

def compressText(text: str, writeDashedText=False) -> bytearray:
    letters: list[str] = generateCompressedText(text)
    
    dataBytes: int = 0
    dataShifted = 3 # 3 bits are reserved for the "number of bits added to the end" header
    for l in letters:
        wasAiCorrect = type(l)==int
        
        if wasAiCorrect:
            # Prepend a 1 to the index. since none of our vocab letters start with a 1 in utf-8 this perfect
            dataBytes <<= 1
            dataShifted += 1
            dataBytes |= 0b1
            
            # huffman tree for the index
            bits, bitsWrote = encodeIndex(l)
            
            dataBytes <<= bitsWrote
            dataShifted += bitsWrote
            dataBytes |= bits
        else:
            dataBytes <<= 8
            dataShifted += 8
            dataBytes |= l.encode("utf-8")[0]
    
    bitsAdded = (8 - (dataShifted % 8))
    dataBytes <<= bitsAdded
    dataShifted += bitsAdded
    
    # write the amount of extra bytes into here
    dataBytes |= (bitsAdded & 0b111) << (dataShifted - 3)
    #print(f"Bits added to end: {bitsAdded} | size: {dataShifted}")
    
    dataByteArray: bytearray = bytearray()
    while dataShifted > 0:
        b = dataBytes & 0xFF
        dataByteArray += bytearray(int.to_bytes(b))
        
        dataBytes >>= 8
        dataShifted -= 8
    dataByteArray = dataByteArray[::-1] # reverse it since we put it in backwards
    
    out: bytearray = dataByteArray
    
    if writeDashedText:
        dashedString = ['-' if type(c) == int else c for c in letters]
        dashedString = ''.join(dashedString)
        
        print(dashedString)
    
    return out

def decompressText(data: bytearray) -> str:
    bitReader = BitReader(data)
    
    extraBitsAtEnd = 0
    for _ in range(0, 3):
        bit = bitReader.readBit()
        extraBitsAtEnd <<= 1
        extraBitsAtEnd |= bit
    
    decompressedText = ""
    llmStates = None
    while True:
        if bitReader.countBitsRemaining() <= extraBitsAtEnd: break # at the end of the data
        
        nextChar = None
        
        # use the AI anyways, even if we're not going to use it's output since we need the state tree to be the same
        if len(decompressedText) > 0: nextChar, llmStates, topPredictedCharacters = generateNextCharacter(decompressedText[-1], llmStates)
        #if len(decompressedText) > 0: nextChar, llmStates, topPredictedCharacters = generateNextCharacter(decompressedText, llmStates)
        
        firstBitRead = bitReader.readBit(advanceCounter=False)
        
        # none of our vocab characters starts with a 1 when encoded in utf-8
        if firstBitRead == 0:
            byte = 0b0
            for i in range(0, 8):
                byte <<= 1
                byte |= bitReader.readBit()
            
            nextChar = bytes([byte]).decode("utf-8")
        elif firstBitRead == 1:
            bitReader.readBit() # discard that first bit since it doesn't get used in our index

            index: int = decodeIndex(bitReader)
            nextChar = topPredictedCharacters[index]
        
        decompressedText += nextChar
    
    print()
    
    return decompressedText


compressed = compressText(stringToEncode, True)
print("Compressed: ", compressed, "\n")

printHistogram()

with open("out.txt", "wb") as f: f.write(compressed)

decompressed = decompressText(compressed)

print(f"Results equal? {stringToEncode==decompressed}")
print(decompressed)

compressedSize = len(compressed)
decompressedSize = len(bytearray(decompressed.encode("utf-8")))

print(compressedSize, decompressedSize, (1-(compressedSize / decompressedSize)), compressedSize < decompressedSize)


