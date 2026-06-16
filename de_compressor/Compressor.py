from LoadModel import generateNextCharacter
from IndexEncoderDecoder import encodeIndex
from BitReaderWriter import BitWriter

def generateCompressedText(stringToEncode: str, oneStepReloaded) -> list[str]:
    states = None
    out = [stringToEncode[0]]

    # 1st character is already in
    for n in range(1, len(stringToEncode)):
        predictedCharacter, states, topPredictedCharacters = generateNextCharacter(oneStepReloaded, stringToEncode[n-1], states)
        correctCharacter = stringToEncode[n]
        
        if correctCharacter in topPredictedCharacters:
            index = topPredictedCharacters.index(correctCharacter)
            out.append(index)
        else:
            # Wrong letter, lets correct it
            out.append(correctCharacter)
    
    return out

def compressText(text: str, oneStepReloaded, writeDashedText=False) -> bytearray:
    letters: list[str] = generateCompressedText(text, oneStepReloaded)
    
    """
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
            dataBytes <<= 1
            dataShifted += 1
            dataBytes |= 0b0
            
            charBytes: bytes = l.encode("utf-8")
            for charByte in charBytes:
                dataBytes <<= 8
                dataShifted += 8
                dataBytes |= charByte
    
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
    """
    
    bw: BitWriter = BitWriter()
    for l in letters:
        wasAiCorrect = type(l)==int
        
        if wasAiCorrect:
            bw.writeBit(0b1) # Prepend a 1 to the index. since none of our vocab letters start with a 1 in utf-8 this perfect
            
            # huffman tree for the index
            bits, bitsWrote = encodeIndex(l)
            bw.writeMultipleBits(bits, bitsWrote)
        else:
            bw.writeBit(0b0)
            charBytes: bytes = l.encode("utf-8")
            for charByte in charBytes: bw.writeMultipleBits(charByte, 8)
    
    out: bytearray = bw.finishWriting()
    
    if writeDashedText:
        dashedString = ['-' if type(c) == int else c for c in letters]
        dashedString = ''.join(dashedString)
        
        print(dashedString)
    
    return out

