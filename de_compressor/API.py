from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
from LoadModel import loadModel, getOneStepGenerator
oneStepReloaded = getOneStepGenerator(loadModel())

import Compressor
import Decompressor

@app.route("/compress", methods=["POST"])
def compressText() -> bytearray:
    data = request.get_json() 
    text: str = data.get("text")
    
    MAX_TEXT_LENGTH = 1500
    
    if len(text) == 0: jsonify({"error": "No text to compress! Length is 0"})
    if len(text) >= MAX_TEXT_LENGTH: jsonify({"error": f"Hihi! In order to keep this API running smoothly we limit each request to {MAX_TEXT_LENGTH} characters\n.Feel free to run it locally from the source code on GitHub to get any length you want <3"})
    
    compressed: bytearray = Compressor.compressText(text, oneStepReloaded, True)
    compressedAsHex: str = ""
    for byte in compressed: compressedAsHex += hex(byte).split("0x")[1].zfill(2)
    
    return jsonify({"data": compressedAsHex})

@app.route("/decompress", methods=["POST"])
def decompressText() -> str:
    data = request.get_json()
    text: str = data.get("data") # formatted as "FE FF 02 AB EF" etc
    isStreaming: bool = data.get("isStreaming", False) # default value is false
    
    if len(text) == 0: jsonify({"error": "No data to decompress! Length is 0"})
    
    try:
        compressed: bytearray = bytearray.fromhex(text)
    except Exception as e:
        return jsonify({"error": f"Failed to turn data into bytes!\n{e}"})
    
    try:
        decompressedStreamGenerator = Decompressor.decompressTextStreaming(compressed, oneStepReloaded)
        if isStreaming:
            return Response(decompressedStreamGenerator, mimetype="text/plain")
        else:
            decompressed = "".join(list(decompressedStreamGenerator)) # collapse all of it into a string, calling list() on a generator returns an array of it's values
            return jsonify({"text": decompressed})
    except IndexError as e:
        return jsonify({"error": f"Decompressor went out of bounds while reading bits. Garbage input data?\n{e}"})
    except UnboundLocalError as e:
        return jsonify({"error": f"Seems like a bad hamming code index. Garbage input data?\n{e}"})
    except Exception as e:
        #raise e
        return jsonify({"error": f"An unknown error occurred while decompressing!\n{e}"})


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
