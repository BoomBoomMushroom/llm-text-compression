from BitReader import BitReader
from LoadModel import generateNextCharacter
from IndexEncoderDecoder import decodeIndex

def decompressText(data: bytearray, oneStepReloaded) -> str:
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
        if len(decompressedText) > 0: nextChar, llmStates, topPredictedCharacters = generateNextCharacter(oneStepReloaded, decompressedText[-1], llmStates)
        
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
