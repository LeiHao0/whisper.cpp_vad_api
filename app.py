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
                                force_reload=False,
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

    # remove_file(filename)
    return f"{output}.txt"

import uuid
from datetime import datetime

def fn_uuid(file):
    print(file.filename)
    filename, file_extension = os.path.splitext(file.filename)

    try:
        parsed_date = datetime.strptime(filename, '%b %d, %Y at %H:%M')
        formatted_date = parsed_date.strftime('%Y-%m-%d_%H-%M')
        print(formatted_date)

    except ValueError:
        print("Unable to parse the date string.")
        formatted_date = str(uuid.uuid4())

    new_filename = f"{formatted_date}{file_extension}"
    return new_filename

def reverse_date(date):
    print('date: ', date)
    date = date.replace('_vad.txt', '')
    try:
        parsed_date = datetime.strptime(date, '%Y-%m-%d_%H-%M')
        formatted_date = parsed_date.strftime('%b %d, %Y at %H:%M')
        return formatted_date
    except ValueError:
        print("Unable to parse the date string.")
        return date

    
import mimetypes


def is_video_audio_file(file):
    mime, _ = mimetypes.guess_type(file.filename)
    print('mime: ', mime)
    return ('audio' in mime) or ('video' in mime)


from pydub import AudioSegment

def conbine_audio(input_files):
    output_file = f'{input_files[0][:10]}.m4a'

    combined_audio = AudioSegment.empty()

    for audio_file in input_files:
        audio = AudioSegment.from_wav(audio_file)
        combined_audio += audio
        combined_audio += AudioSegment.silent(duration=1000)

    combined_audio.export(output_file, format="mp4")

    for audio_file in input_files:
        os.remove(audio_file)
        print("removed: ", audio_file)

    print("Audio files combined and compressed to", output_file)

    return output_file


def conbine_text(input_files):
    output_file = f'{input_files[0][:10]}.txt'

    with open(output_file, 'w') as outfile:
        for file_name in input_files:
            outfile.write(f'\n## {reverse_date(file_name)}\n\n')
            with open(file_name, 'r') as infile:
                content = infile.read()
                outfile.write(content)

    for f in input_files:
        os.remove(f)
        print("removed: ", f)

    print("Text files combined and saved to", output_file)
    return output_file

import shutil
import zipfile

def zip(combined_text, combined_audio):
    zip_file = f'{remove_ext(combined_text)}.zip'
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(combined_text)
        zf.write(combined_audio)
    
    os.remove(combined_text)
    os.remove(combined_audio)

    return zip_file
    

from flask import Flask, render_template, request, send_file, jsonify, Response

app = Flask(__name__)



@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file')
        fns_vad = []
        fns_txt = []
        for file in uploaded_files :
            if file and is_video_audio_file(file):
                fn = fn_uuid(file)
                file.save(fn)
                
                fn_vad = vad(convert(fn))
                fn_txt = tts(fn_vad)
                fns_vad.append(fn_vad)
                fns_txt.append(fn_txt)
            else:
                print(file, 'is not audio or video')

        combined_text = conbine_text(fns_txt)
        combined_audio = conbine_audio(fns_vad)
        zip_file = zip(combined_text, combined_audio)

        return send_file(zip_file, as_attachment=True, download_name=zip_file)



if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')