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
    
    def readByte(self, advanceCounter: bool=True) -> int:
        byte = 0b0
        for i in range(0, 8):
            byte <<= 1
            byte |= self.readBit()
        if advanceCounter == False: self.bitPos -= 8
        return byte

class BitWriter:
    def __init__(self):
        self.dataBytes: int = 0
        self.dataShifted: int = 3 # 3 bits are reserved for the "number of bits added to the end" header
        self.isDoneWriting: bool = False
        self.result: bytearray = bytearray()
    
    def getResult(self) -> bytearray:
        if len(self.result) == 0:
            self.result = bytearray()
            while self.dataShifted > 0:
                b = self.dataBytes & 0xFF
                self.result += bytearray(int.to_bytes(b))
                
                self.dataBytes >>= 8
                self.dataShifted -= 8
            self.result = self.result[::-1] # reverse it since we put it in backwards
        
        return self.result
    
    def finishWriting(self) -> bytearray:
        if self.isDoneWriting: return
        
        # Add padding
        bitsAdded = (8 - (self.dataShifted % 8))
        self.writeMultipleBits(0, bitsAdded) # write a few 0s to the end to align it to a byte
        
        # Write the header
        self.dataBytes |= (bitsAdded & 0b111) << (self.dataShifted - 3)
        
        self.isDoneWriting = True
        return self.getResult()
    
    def writeBit(self, bit):
        if self.isDoneWriting: return
        
        self.dataBytes <<= 1
        self.dataShifted += 1
        self.dataBytes |= (bit & 0b1)
    
    def writeMultipleBits(self, bits, bitsToWrite):
        if self.isDoneWriting: return
        mask = (1 << bitsToWrite) - 1 # generates a mask that is an N amount of 1s (ex. N=5 ->  0b11111)

        self.dataBytes <<= bitsToWrite
        self.dataShifted += bitsToWrite
        self.dataBytes |= (bits & mask)

