from BitReaderWriter import BitReader

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
