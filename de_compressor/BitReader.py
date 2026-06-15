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
