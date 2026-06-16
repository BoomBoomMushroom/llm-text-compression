from BitReaderWriter import BitReader
from LoadModel import generateNextCharacter
from IndexEncoderDecoder import decodeIndex

def decompressTextStreaming(data: bytearray, oneStepReloaded):
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
        
        #firstBitRead = bitReader.readBit(advanceCounter=False)
        firstBitRead = bitReader.readBit()
        
        # none of our vocab characters starts with a 1 when encoded in utf-8
        if firstBitRead == 0:
            # if we have a special character, like "🐟"
            bytesToRead = 0
            firstByte = bitReader.readByte(advanceCounter=False)
            if firstByte & 0b10000000 == 0: bytesToRead = 1
            if firstByte & 0b11100000 == 0b110_00000: bytesToRead = 2
            if firstByte & 0b11110000 == 0b1110_0000: bytesToRead = 3
            if firstByte & 0b11111000 == 0b11110_000: bytesToRead = 4
            
            bytesRead = []
            for i in range(bytesToRead): bytesRead.append(bitReader.readByte())
            
            nextChar = bytes(bytesRead).decode("utf-8")
        elif firstBitRead == 1:
            #bitReader.readBit() # discard that first bit since it doesn't get used in our index

            index: int = decodeIndex(bitReader)
            nextChar = topPredictedCharacters[index]
        
        decompressedText += nextChar
        yield nextChar
    
    # no return

def decompressText(data: bytearray, oneStepReloaded) -> str:
    return "".join(decompressTextStreaming(data, oneStepReloaded))

