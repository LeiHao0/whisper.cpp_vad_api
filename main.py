# conda activate py310-whisper

import torch
from IPython.display import Audio
from pprint import pprint
import os


import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Command failed with error: {result.stderr.strip()}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

def remove_ext(filename):
    return os.path.splitext(filename)[0]

def convert(filename):
    output=f"{remove_ext(filename)}.wav"

    remove_file(output)

    cmd = f'ffmpeg -i {filename} -ar 16000 -ac 1 -c:a pcm_s16le {output}'
    run_command(cmd)

    remove_file(filename)
    return output

def vad(filename):

    SAMPLING_RATE = 16000
    torch.set_num_threads(10)

    USE_ONNX = True
    
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True,
                                onnx=USE_ONNX)

    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

    wav = read_audio(filename, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    pprint(speech_timestamps)

    output=f"{remove_ext(filename)}_vad.wav"

    save_audio(output,
    collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
    
    remove_file(filename)
    return output


def tts(filename):
    output = remove_ext(filename)
    cmd = f"""../main -m ../models/ggml-large.bin \
-f {filename} \
-of {output} \
-t 20 \
-l 'auto' \
-otxt \
"""
    run_command(cmd)

    remove_file(filename)
    return f"{output}.txt"

def format_fn(fn):
    return fn.replace(" ", "_")

from flask import Flask, render_template, request, send_file, jsonify

app = Flask(__name__)

# Serve the HTML form on the root URL
@app.route('/', methods=['GET'])
def serve_form():
    return render_template('upload_form.html')

@app.route('/api', methods=['POST'])
def api():
    file = request.files['file']
    if file:
        fn = format_fn(file.filename)
        file.save(fn)
        output = tts(convert(fn))
        txt = open(output, 'r').read()
        print(txt)
        response_data = {'text': txt}

        remove_file(output)
        return jsonify(response_data), 200
    else:
        return jsonify({'error': 'No file provided'}), 400

# Handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        fn = format_fn(file.filename)
        file.save(fn)
        output = tts(vad(convert(fn)))
        f = send_file(f'{output}', as_attachment=True)
        remove_file(output)
        return f
    else:
        return 'No file provided'

if __name__ == '__main__':
    app.run(debug=True)
