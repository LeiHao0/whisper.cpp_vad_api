# Whisper.cpp VAD API

This repository contains a Python script that demonstrates the usage of voice activity detection (VAD) using the Silero VAD model. The script allows for audio file processing, VAD application, and text-to-speech (TTS) conversion.

## Prerequisites

Ensure you have Python environment set up using conda:

```bash
conda activate py310-whisper
```

## Usage

1. Run the script and utilize the provided functions to process audio files, apply voice activity detection, and convert text to speech.

2. The following main functions are available:

   - `convert(filename)`: Convert an audio file to the required format for VAD using ffmpeg.
   
   - `vad(filename)`: Apply Voice Activity Detection (VAD) to the audio file using the Silero VAD model.
   
   - `tts(filename)`: Perform Text-to-Speech (TTS) on the audio file.

3. Utilize the provided Flask API to interact with the script:

   - Access the root URL (`/`) to serve the HTML upload form.
   
   - Upload an audio file using the provided form, and the API will process the audio, apply VAD, and perform TTS, returning the generated text.

   - Alternatively, you can directly send an audio file to the `/upload` endpoint to receive the processed audio file with VAD and TTS applied.

For more details and examples, refer to the provided code in `whisper.cpp_vad_api.py`.

## Running the Application

To run the application locally, execute:

```bash
python main.py
```

Ensure you have the necessary dependencies installed and an appropriate Python environment activated. The application will start a local server, allowing interaction with the provided API endpoints.