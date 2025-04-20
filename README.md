# Whisper Audio Transcriber

Audio transcription tool using OpenAI's Whisper model.

## Features
- Multiple language support (English/Chinese/Auto)
- SRT/TXT/VTT format outputs
- Edit transcripts before download
- Paginated view for long transcripts

## Setup
```bash
git clone https://github.com/zhuotao25/whisper-transcriber-ui.git
cd whisper-transcirber-ui
pip3 install -r requirements.txt
streamlit run app.py --server.enableXsrfProtection false