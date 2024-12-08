from flask import Flask, request, jsonify, abort
from indexing_pipeline import IndexingPipeline
import audio_to_text 
from rag_pipeline import RagPipeline
import os

app = Flask(__name__)


def remove_file(file):
    try:
        os.remove(file)
    except:
        pass

@app.route('/', methods = ['GET'])
def home():
    return "Hello World"

@app.route("/add_audio_files", methods = ['POST'])
def add_audio_files():
    try:
        if 'file' not in request.files:
            return jsonify({"response": "No File"})
        file = request.files['file']
        file_path = os.path.join(os.getcwd(), file.filename)
        with open(file_path, 'wb') as f:
            f.write(file.read())
        print(file_path)
        output = audio_to_text.query(file_path)

        with open('temporary_file.txt', 'w') as f1:
            f1.write(output['text'])

        indexing_pipeline = IndexingPipeline()
        indexing_pipeline.build_indexing_pipeline()
        indexing_pipeline.run_indexing_pipeline(['temporary_file.txt'])

        remove_file(f)
        remove_file(f1)

        return jsonify({'response': "Added file to the vector database"})
    
    except Exception as exc:
        return str(exc), 500

@app.route("/test_audio_model", methods = ['POST'])
def test_audio_model():
    try:
        if 'file' not in request.files:
            return jsonify({"response": "No File"})
        file = request.files['file']
        print(file)
        file_path = os.path.join(os.getcwd(), file.filename)
        with open(file_path, 'wb') as f:
            f.write(file.read())
        
        print(file_path)
        output = audio_to_text.query(file_path)
        print(output)
        text = output['text']

        remove_file(f)
        return jsonify({"response": text})

    except Exception as exc:
        return str(exc), 500

@app.route("/recommendation", methods = ['POST'])
def recommendation():
    try:
        data = request.json
        query = data['query']
        rag_pipeline = RagPipeline()
        rag_pipeline.build_rag_pipeline()
        answer = rag_pipeline.run_rag_pipeline(query)
        return jsonify({'answer': answer})
    
    except Exception as exc:
        return str(exc), 500

if __name__ == '__main__':
    app.run(debug=True)