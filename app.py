import streamlit as st
import whisper
from tempfile import NamedTemporaryFile
import os
import time

# Set page config
st.set_page_config(page_title="Whisper Transcription", layout="wide")

# Title and description
st.title("🎤 Whisper Audio Transcription")
st.write("Powered by OpenAI's Whisper model")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=3,  # Default to medium
        help="Larger models are more accurate but slower"
    )
    
    language_option = st.radio(
        "Language",
        ["Automatic Detection", "English", "Chinese"],
        index=0,
        help="Select language or auto-detect"
    )
    
    format_option = st.selectbox(
        "Output Format",
        ["SRT", "TXT", "VTT"],
        index=0,
        help="Select output file format"
    )

# Language mapping dictionary
language_mapping = {
    "Automatic Detection": None,
    "English": "en",
    "Chinese": "zh"
}

def format_time(seconds, format_type):
    """Convert seconds to SRT/VTT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if format_type == "SRT":
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    elif format_type == "VTT":
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return ""

# File upload section
st.subheader("Upload Audio File")
audio_file = st.file_uploader(
    "Choose an audio file", 
    type=["wav", "mp3", "ogg", "m4a"],
    help="Supported formats: WAV, MP3, OGG, M4A"
)

if audio_file is not None:
    # Display audio player
    st.audio(audio_file, format="audio/wav")
    
    # Transcription button
    if st.button("Transcribe Audio"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Loading Whisper model..."):
            try:
                model = whisper.load_model(model_option)
                progress_bar.progress(30)
                status_text.text("Model loaded successfully")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()

        # Save uploaded file to temporary file
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file.read())
            temp_path = temp_file.name

        # Perform transcription
        try:
            progress_bar.progress(40)
            status_text.text("Starting transcription...")
            
            result = model.transcribe(
                temp_path,
                language=language_mapping[language_option],
                verbose=True
            )
            
            progress_bar.progress(100)
            status_text.text("Transcription complete!")
            time.sleep(0.5)

            # Generate formatted output
            output_text = ""
            if format_option == "TXT":
                output_text = result["text"]
            else:
                for i, segment in enumerate(result["segments"], start=1):
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    
                    if format_option == "SRT":
                        output_text += f"{i}\n"
                        output_text += f"{format_time(start, 'SRT')} --> {format_time(end, 'SRT')}\n"
                        output_text += f"{text}\n\n"
                    elif format_option == "VTT":
                        output_text += f"{format_time(start, 'VTT')} --> {format_time(end, 'VTT')}\n"
                        output_text += f"{text}\n\n"

            # Initialize session state for pagination
            chunk_size = 2000  # Characters per page
            st.session_state.pages = [output_text[i:i+chunk_size] 
                                    for i in range(0, len(output_text), chunk_size)]
            st.session_state.current_page = 0
            st.session_state.edited_pages = st.session_state.pages.copy()

            # Display results
            st.success("✅ Transcription Complete!")
            
            # Show detected language if auto-detection was used
            if language_option == "Automatic Detection":
                st.write(f"**Detected Language:** {result['language'].title()}")

        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
        finally:
            os.unlink(temp_path)
            progress_bar.empty()
            status_text.empty()

# Edit and pagination controls
if 'pages' in st.session_state and len(st.session_state.pages) > 0:
    st.subheader("Edit Transcription")
    
    # Page navigation
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.session_state.current_page > 0:
            if st.button("← Previous"):
                st.session_state.current_page -= 1
    with col2:
        if st.session_state.current_page < len(st.session_state.pages) - 1:
            if st.button("Next →"):
                st.session_state.current_page += 1
                
    current_page = st.session_state.current_page
    st.write(f"Page {current_page + 1} of {len(st.session_state.pages)}")
    
    # Editable text area with persistent edits
    edited_text = st.text_area(
        "Edit transcript (changes are auto-saved)",
        value=st.session_state.edited_pages[current_page],
        height=300,
        key=f"editor_{current_page}"
    )
    
    # Update edited pages
    st.session_state.edited_pages[current_page] = edited_text
    
    # Combine all pages for download
    final_transcript = "".join(st.session_state.edited_pages)
    
    # Download button
    file_extension = format_option.lower()
    st.download_button(
        "Download Edited Transcript",
        final_transcript,
        file_name=f"edited_transcript.{file_extension}",
        mime="text/plain"
    )

# Add some instructions
st.markdown("""
---
### Instructions:
1. Select model size in the sidebar (medium recommended)
2. Choose language or use auto-detection
3. Upload an audio file (max 200MB)
4. Click 'Transcribe Audio' button
5. Review and edit transcript using pagination
6. Download final edited version
""")

# Add footer
st.markdown("""
---
*Built with OpenAI Whisper and Streamlit*  
*Support formats: WAV, MP3, OGG, M4A*
""")