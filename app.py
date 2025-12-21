"""
Flask web application for Sonemote - Emotion-Based Music Player
Provides web interface with camera feed and music management.
"""

import os
import sys
import cv2
import numpy as np
import base64
import re
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from tensorflow import keras
from pathlib import Path
import json
import yt_dlp

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models import create_emotion_model

app = Flask(__name__)
CORS(app)

# Global variables
model = None
face_cascade = None
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Emotion to folder mapping
EMOTION_FOLDERS = {
    'Angry': 'angry',
    'Disgust': 'disgust',
    'Fear': 'fear',
    'Happy': 'happy',
    'Sad': 'sad',
    'Surprise': 'surprise',
    'Neutral': 'neutral'
}


def init_model(model_path='emotion_model.h5'):
    """Initialize the emotion detection model."""
    global model, face_cascade
    
    try:
        # Load model
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"WARNING: Model not found at {model_path}. Creating untrained model structure.")
            model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
            print("Untrained model structure created.")
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        print(f"Looking for face cascade at: {cascade_path}")
        
        if not os.path.exists(cascade_path):
            # Try alternative path
            alt_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            if os.path.exists(alt_path):
                cascade_path = alt_path
                print(f"Using alternative cascade path: {cascade_path}")
            else:
                raise FileNotFoundError(f"Face cascade not found at {cascade_path} or {alt_path}")
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Verify cascade loaded correctly
        if face_cascade.empty():
            raise ValueError(f"Failed to load face cascade from {cascade_path}")
        
        print("Model and face cascade loaded successfully!")
        
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        raise


def preprocess_face(face_roi, target_size=(48, 48)):
    """Preprocess face ROI for model prediction."""
    # Convert BGR to grayscale if needed
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Reshape to (1, 48, 48, 1) for model input
    processed = normalized.reshape(1, target_size[1], target_size[0], 1)
    
    return processed


def apply_emotion_weights(predictions):
    """
    Apply weighting to emotion predictions to favor harder-to-detect emotions.
    Increases weights for: Disgust
    Decreases weight for: Neutral
    
    Args:
        predictions: Raw prediction array (1, 7) or (7,)
        
    Returns:
        Weighted predictions (same shape as input)
    """
    # Emotion order: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Weight multipliers: increase Disgust (index 1), decrease Neutral (index 6)
    weight_multipliers = np.array([0.8, 3.0, 1.5, 1.0, 1.8, 1.0, 0.3])  # Boost Disgust significantly, reduce Neutral
    
    # Ensure predictions is numpy array
    predictions = np.array(predictions)
    original_shape = predictions.shape
    
    # Flatten if needed for easier processing
    if predictions.ndim == 2:
        predictions = predictions[0]  # Take first (and likely only) row
    
    # Apply weights
    weighted = predictions * weight_multipliers
    
    # Renormalize to maintain probability distribution
    weighted = weighted / weighted.sum()
    
    # Reshape back to original shape
    if len(original_shape) == 2:
        weighted = weighted.reshape(1, -1)
    
    return weighted


def detect_emotion_from_image(image_data):
    """
    Detect emotion from base64 encoded image.
    
    Args:
        image_data: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        dict with emotion, confidence, and face_detected
    """
    global model, face_cascade
    
    try:
        # Ensure model and face_cascade are initialized
        if model is None or face_cascade is None:
            print("Model or face_cascade not initialized. Initializing now...")
            init_model()
            if model is None or face_cascade is None:
                return {'error': 'Failed to initialize model or face cascade', 'face_detected': False}
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'error': 'Could not decode image', 'face_detected': False}
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {
                'emotion': None,
                'confidence': 0.0,
                'face_detected': False
            }
        
        # Process first face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess and predict
        processed_face = preprocess_face(face_roi)
        predictions = model.predict(processed_face, verbose=0)
        
        # Apply weighting to favor harder-to-detect emotions (Disgust) and reduce Neutral
        predictions = apply_emotion_weights(predictions)
        
        # Get emotion with highest confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = emotion_labels[emotion_idx]
        
        # Get all emotion probabilities
        emotion_probs = {
            emotion_labels[i]: float(predictions[0][i])
            for i in range(len(emotion_labels))
        }
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'face_detected': True,
            'emotion_probs': emotion_probs,
            'face_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        }
    
    except Exception as e:
        return {'error': str(e), 'face_detected': False}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    """API endpoint for emotion detection."""
    try:
        # Ensure model is initialized
        if model is None or face_cascade is None:
            print("Model not initialized in detect_emotion endpoint. Initializing...")
            init_model()
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        result = detect_emotion_from_image(image_data)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in detect_emotion endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint to verify model and face cascade are loaded."""
    status = {
        'status': 'ok',
        'model_loaded': model is not None,
        'face_cascade_loaded': face_cascade is not None,
        'model_path_exists': os.path.exists('emotion_model.h5')
    }
    
    if face_cascade is not None:
        status['face_cascade_empty'] = face_cascade.empty()
    
    return jsonify(status)


@app.route('/api/music/<emotion>', methods=['GET'])
def get_music_list(emotion):
    """Get list of music files for an emotion category."""
    folder_name = EMOTION_FOLDERS.get(emotion)
    if not folder_name:
        return jsonify({'error': 'Invalid emotion'}), 400
    
    music_dir = Path('music') / folder_name
    if not music_dir.exists():
        return jsonify({'songs': []})
    
    # Get all audio files
    audio_formats = ('.mp3', '.wav', '.ogg', '.m4a', '.flac')
    songs = []
    for file_path in music_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_formats:
            songs.append({
                'name': file_path.stem,
                'filename': file_path.name,
                'path': f'/api/music/file/{emotion}/{file_path.name}'
            })
    
    return jsonify({'songs': songs})


@app.route('/api/music/file/<emotion>/<filename>')
def serve_music_file(emotion, filename):
    """Serve a music file."""
    folder_name = EMOTION_FOLDERS.get(emotion)
    if not folder_name:
        return jsonify({'error': 'Invalid emotion'}), 400
    
    file_path = Path('music') / folder_name / filename
    if not file_path.exists() or not file_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(str(file_path))


@app.route('/api/music/<emotion>', methods=['POST'])
def add_music(emotion):
    """Add a music file to an emotion category."""
    folder_name = EMOTION_FOLDERS.get(emotion)
    if not folder_name:
        return jsonify({'error': 'Invalid emotion'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if it's an audio file
    audio_formats = ('.mp3', '.wav', '.ogg', '.m4a', '.flac')
    if not file.filename.lower().endswith(audio_formats):
        return jsonify({'error': 'Invalid file format. Only audio files are allowed.'}), 400
    
    # Create directory if it doesn't exist
    music_dir = Path('music') / folder_name
    music_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = music_dir / file.filename
    file.save(str(file_path))
    
    return jsonify({
        'success': True,
        'message': f'File {file.filename} added to {emotion} category',
        'filename': file.filename
    })


@app.route('/api/music/<emotion>/<filename>', methods=['DELETE'])
def delete_music(emotion, filename):
    """Delete a music file from an emotion category."""
    folder_name = EMOTION_FOLDERS.get(emotion)
    if not folder_name:
        return jsonify({'error': 'Invalid emotion'}), 400
    
    file_path = Path('music') / folder_name / filename
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        file_path.unlink()
        return jsonify({'success': True, 'message': f'File {filename} deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get list of available emotions."""
    return jsonify({'emotions': list(emotion_labels.values())})


@app.route('/api/youtube/search', methods=['GET'])
def search_youtube():
    """Search YouTube for videos."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'default_search': 'ytsearch',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for videos (yt-dlp handles ytsearch: automatically)
            search_query = f'ytsearch10:{query}'
            info = ydl.extract_info(search_query, download=False)
            
            results = []
            if 'entries' in info:
                for entry in info['entries']:
                    if entry and entry.get('id'):
                        results.append({
                            'id': entry.get('id', ''),
                            'title': entry.get('title', 'Unknown'),
                            'url': entry.get('url', f"https://www.youtube.com/watch?v={entry.get('id', '')}"),
                            'duration': entry.get('duration', 0),
                            'thumbnail': entry.get('thumbnail', entry.get('thumbnails', [{}])[0].get('url', '') if entry.get('thumbnails') else '')
                        })
            
            return jsonify({'results': results[:10]})  # Limit to 10 results
    
    except Exception as e:
        import traceback
        print(f"YouTube search error: {traceback.format_exc()}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


def check_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available."""
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')
    
    # Check environment variable for custom ffmpeg location
    ffmpeg_location = os.environ.get('FFMPEG_LOCATION')
    if ffmpeg_location:
        ffmpeg_path = ffmpeg_location if os.path.exists(ffmpeg_location) else None
        # ffprobe is usually in the same directory as ffmpeg
        if ffmpeg_path:
            ffprobe_dir = os.path.dirname(ffmpeg_path)
            ffprobe_candidate = os.path.join(ffprobe_dir, 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe')
            if os.path.exists(ffprobe_candidate):
                ffprobe_path = ffprobe_candidate
    
    return ffmpeg_path is not None and ffprobe_path is not None, ffmpeg_path


@app.route('/api/youtube/download', methods=['POST'])
def download_youtube():
    """Download audio from YouTube URL and add to emotion category."""
    try:
        data = request.get_json()
        url = data.get('url')
        emotion = data.get('emotion')
        
        if not url or not emotion:
            return jsonify({'error': 'URL and emotion are required'}), 400
        
        folder_name = EMOTION_FOLDERS.get(emotion)
        if not folder_name:
            return jsonify({'error': 'Invalid emotion'}), 400
        
        # Check if FFmpeg is available
        ffmpeg_available, ffmpeg_path = check_ffmpeg_available()
        if not ffmpeg_available:
            return jsonify({
                'error': 'FFmpeg and ffprobe are required for YouTube audio downloads. Please install FFmpeg or set FFMPEG_LOCATION environment variable.'
            }), 400
        
        # Create directory if it doesn't exist
        music_dir = Path('music') / folder_name
        music_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info first to get the title
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
        
        # Clean filename to remove invalid characters
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        # Limit filename length
        safe_title = safe_title[:100] if len(safe_title) > 100 else safe_title
        
        # Configure yt-dlp for audio extraction with cleaned title
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(music_dir / f'{safe_title}.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        
        # Only add FFmpeg postprocessor if ffmpeg is available
        if ffmpeg_available:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
            # Set ffmpeg location if provided via environment variable
            if ffmpeg_path:
                ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)
        
        # Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded file (postprocessor should have converted to mp3)
        filename = f"{safe_title}.mp3"
        downloaded_file = music_dir / filename
        
        # If mp3 doesn't exist, look for the file with the original extension before conversion
        if not downloaded_file.exists():
            # Check for files that match the title pattern
            for file_path in music_dir.glob(f"{safe_title}.*"):
                if file_path.suffix.lower() in ['.m4a', '.webm', '.opus', '.ogg', '.mp3']:
                    if file_path.suffix.lower() != '.mp3':
                        # If ffmpeg wasn't available, keep the original format
                        if not ffmpeg_available:
                            downloaded_file = file_path
                            filename = file_path.name
                        else:
                            # Rename to mp3 if it's not already
                            downloaded_file = file_path.with_suffix('.mp3')
                            file_path.rename(downloaded_file)
                    else:
                        downloaded_file = file_path
                    break
        
        if not downloaded_file.exists():
            raise Exception(f"Downloaded file not found. Please check if FFmpeg is installed and the download completed successfully.")
        
        return jsonify({
            'success': True,
            'message': f'Audio downloaded and added to {emotion} category',
            'filename': filename,
            'title': title
        })
    
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'ffprobe' in error_msg.lower() or 'ffmpeg' in error_msg.lower():
            return jsonify({
                'error': 'FFmpeg/ffprobe not found. Please install FFmpeg or set FFMPEG_LOCATION environment variable pointing to the FFmpeg directory.'
            }), 500
        return jsonify({'error': f'Download failed: {error_msg}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Initialize model when module is imported (works with gunicorn too)
# This ensures the model is loaded before any requests are handled
try:
    init_model()
except Exception as e:
    print(f"WARNING: Failed to initialize model on startup: {e}")
    print("Model will be initialized on first request.")


if __name__ == '__main__':
    # Get port and host from environment variables (for production deployment)
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Run Flask app
    print("\n" + "="*60)
    print("Sonemote Web App Starting...")
    print("="*60)
    print(f"Open your browser and navigate to: http://localhost:{port}")
    if host == '0.0.0.0':
        print(f"Or access from network at: http://<your-ip>:{port}")
    print("="*60 + "\n")
    
    app.run(debug=debug, host=host, port=port, threaded=True)

