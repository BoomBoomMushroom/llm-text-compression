stringToEncode = "coda started making their obelisks and then theo made a lavacast and now those are the \"Rockets\" space X makes"
#stringToEncode = input("Enter string to encode: ")

#stringToEncode = """I have the first regular season chess match tomorrow after school against Madeira, and I'm just so happy to be playing (on an important board), after all of the effort I put in and just barely missing the top boards Sophomore and Junior year after all the work I put in, I was genuinely quite disappointed last year, but I at least got to be a starting board in the end-of season tournament where we got first (Not the playoffs). But really I felt like there was definitely a chance that that would be the only like direct showing of my efforts to improve all this much and make the team. Wyoming just has really good chess players bro but like I did it this year and I'm going to try my best to lock tf in this year."""

# Load Model

from LoadModel import loadModel, getOneStepGenerator
oneStepReloaded = getOneStepGenerator(loadModel())

# Actually compress and decompress it and stuff

import Compressor
import Decompressor

compressed: bytearray = Compressor.compressText(stringToEncode, oneStepReloaded, True)
print("Compressed: ", compressed, "\n")
with open("out.txt", "wb") as f: f.write(compressed)

decompressed: str = Decompressor.decompressText(compressed, oneStepReloaded)

print(f"Results equal? {stringToEncode==decompressed}")
print(decompressed)

compressedSize = len(compressed)
decompressedSize = len(bytearray(decompressed.encode("utf-8")))

print(compressedSize, decompressedSize, (1-(compressedSize / decompressedSize)), compressedSize < decompressedSize)


