"""
Flask web application for Sonemote - Emotion-Based Music Player
Provides web interface with camera feed and music management.
"""

import os
import sys

# Force TensorFlow to use CPU only (prevents CUDA errors on Railway)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import cv2
import numpy as np
import base64
import re
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Import TensorFlow after setting environment variables
import tensorflow as tf
# Configure TensorFlow to use CPU and reduce memory
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Limit TensorFlow memory to prevent OOM errors
try:
    # Set memory growth for CPU (if supported)
    physical_devices = tf.config.list_physical_devices('CPU')
    # Limit memory allocation
    tf.config.experimental.set_memory_growth(physical_devices[0], True) if physical_devices else None
except:
    pass

# Set TensorFlow to use less memory
try:
    # Limit inter-op and intra-op parallelism to reduce memory
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except:
    pass

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
        # Load model with memory optimization
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            # Load model with compile=False to reduce memory usage
            # We'll compile only when needed
            try:
                model = keras.models.load_model(model_path, compile=False)
                # Compile with minimal settings to reduce memory
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("Model loaded successfully!")
            except Exception as load_error:
                print(f"Error loading model: {load_error}")
                # Try loading without compilation
                model = keras.models.load_model(model_path, compile=False)
                print("Model loaded without compilation (will compile on first use)")
        else:
            print(f"WARNING: Model not found at {model_path}. Creating untrained model structure.")
            model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
            print("Untrained model structure created.")
        
        # Load face cascade - try multiple methods
        cascade_path = None
        face_cascade = None
        
        # Method 1: Try OpenCV's built-in path (most common)
        try:
            default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            print(f"Trying default cascade path: {default_path}")
            face_cascade = cv2.CascadeClassifier(default_path)
            if not face_cascade.empty():
                cascade_path = default_path
                print("Successfully loaded cascade from default path")
        except Exception as e:
            print(f"Default path failed: {e}")
        
        # Method 2: Try alternative system paths if default failed
        if face_cascade is None or face_cascade.empty():
            alt_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            for alt_path in alt_paths:
                try:
                    print(f"Trying alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        test_cascade = cv2.CascadeClassifier(alt_path)
                        if not test_cascade.empty():
                            face_cascade = test_cascade
                            cascade_path = alt_path
                            print(f"Successfully loaded cascade from: {cascade_path}")
                            break
                except Exception as e:
                    print(f"Alternative path {alt_path} failed: {e}")
                    continue
        
        # Method 3: Search in Python package locations
        if face_cascade is None or face_cascade.empty():
            try:
                import site
                for site_package in site.getsitepackages():
                    test_path = os.path.join(site_package, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
                    if os.path.exists(test_path):
                        print(f"Trying package path: {test_path}")
                        test_cascade = cv2.CascadeClassifier(test_path)
                        if not test_cascade.empty():
                            face_cascade = test_cascade
                            cascade_path = test_path
                            print(f"Successfully loaded cascade from package: {cascade_path}")
                            break
            except Exception as e:
                print(f"Package search failed: {e}")
        
        # Final check
        if face_cascade is None or face_cascade.empty():
            error_msg = (
                f"Failed to load face cascade classifier. Tried:\n"
                f"  - {cv2.data.haarcascades}haarcascade_frontalface_default.xml\n"
                f"  - /usr/share/opencv4/haarcascades/\n"
                f"  - /usr/local/share/opencv4/haarcascades/\n"
                f"  - /usr/share/opencv/haarcascades/\n"
                f"  - Python package locations\n\n"
                f"Please ensure opencv-python-headless is properly installed."
            )
            raise FileNotFoundError(error_msg)
        
        print("Model and face cascade loaded successfully!")
        
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        import traceback
        traceback.print_exc()
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
            try:
                init_model()
            except Exception as e:
                print(f"Error during lazy initialization: {e}")
                import traceback
                traceback.print_exc()
                return {'error': f'Failed to initialize model or face cascade: {str(e)}', 'face_detected': False}
            
            if model is None or face_cascade is None:
                return {'error': 'Failed to initialize model or face cascade', 'face_detected': False}
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            print(f"DEBUG: Decoded image bytes, size: {len(image_bytes)} bytes")
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("ERROR: cv2.imdecode returned None - image format may be invalid")
                # Try to get more info about the image
                print(f"DEBUG: Image bytes length: {len(image_bytes)}")
                print(f"DEBUG: First 20 bytes (hex): {image_bytes[:20].hex()}")
                # Check if it's a valid image format
                if image_bytes[:4] == b'\x89PNG':
                    print("DEBUG: Detected PNG format")
                elif image_bytes[:2] == b'\xff\xd8':
                    print("DEBUG: Detected JPEG format")
                else:
                    print(f"DEBUG: Unknown image format, first bytes: {image_bytes[:10]}")
                return {'error': 'Could not decode image - invalid format', 'face_detected': False}
            
            print(f"DEBUG: Successfully decoded image, shape: {frame.shape}")
        except Exception as decode_error:
            print(f"ERROR decoding image: {decode_error}")
            import traceback
            traceback.print_exc()
            return {'error': f'Image decode error: {str(decode_error)}', 'face_detected': False}
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces - ensure face_cascade is valid
        if face_cascade is None:
            print("ERROR: face_cascade is None in detect_emotion_from_image")
            return {'error': 'Face cascade not initialized', 'face_detected': False}
        
        # Log image info for debugging
        print(f"DEBUG: Image shape: {frame.shape}, Grayscale shape: {gray.shape}")
        print(f"DEBUG: Face cascade empty check: {face_cascade.empty()}")
        print(f"DEBUG: Image dimensions - width: {frame.shape[1]}, height: {frame.shape[0]}")
        
        # Check if image is too small
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            print(f"WARNING: Image is very small: {frame.shape}")
        
        # Check image quality
        gray_mean = gray.mean()
        gray_std = gray.std()
        print(f"DEBUG: Grayscale stats - mean: {gray_mean:.2f}, std: {gray_std:.2f}, min: {gray.min()}, max: {gray.max()}")
        
        # Warn if image is too dark or too bright (might affect detection)
        if gray_mean < 30:
            print("WARNING: Image appears very dark (mean < 30)")
        elif gray_mean > 225:
            print("WARNING: Image appears very bright (mean > 225)")
        
        try:
            # Use more lenient parameters for better detection
            # Lower scaleFactor = more thorough search (slower but better)
            # Lower minNeighbors = more detections (may include false positives)
            # Smaller minSize = detect smaller faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More thorough (was 1.1)
                minNeighbors=3,     # More lenient (was 5)
                minSize=(20, 20),   # Smaller faces (was 30, 30)
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"DEBUG: Detected {len(faces)} face(s)")
        except Exception as e:
            print(f"ERROR in face detection: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Face detection error: {str(e)}', 'face_detected': False}
        
        if len(faces) == 0:
            print("DEBUG: No faces detected. Image stats - min:", gray.min(), "max:", gray.max(), "mean:", gray.mean())
            # Try with even more lenient parameters as fallback
            try:
                print("DEBUG: Trying fallback detection with more lenient parameters...")
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.03,
                    minNeighbors=2,
                    minSize=(15, 15),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                print(f"DEBUG: Fallback detection found {len(faces)} face(s)")
            except Exception as e2:
                print(f"DEBUG: Fallback detection also failed: {e2}")
            
            if len(faces) == 0:
                return {
                    'emotion': None,
                    'confidence': 0.0,
                    'face_detected': False,
                    'debug_info': {
                        'image_shape': frame.shape,
                        'gray_shape': gray.shape,
                        'gray_stats': {
                            'min': float(gray.min()),
                            'max': float(gray.max()),
                            'mean': float(gray.mean())
                        }
                    }
                }
        
        # Process first face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess and predict
        processed_face = preprocess_face(face_roi)
        
        # Use predict with minimal memory footprint
        try:
            # Use __call__ instead of predict for lower memory usage
            # This avoids creating a new computation graph
            with tf.device('/CPU:0'):  # Explicitly use CPU
                predictions = model(processed_face, training=False)
                # Convert to numpy if needed
                if hasattr(predictions, 'numpy'):
                    predictions = predictions.numpy()
                else:
                    predictions = np.array(predictions)
        except Exception as predict_error:
            print(f"ERROR in model prediction: {predict_error}")
            import traceback
            traceback.print_exc()
            # If prediction fails, return error but keep face_detected=True
            return {
                'emotion': None,
                'confidence': 0.0,
                'face_detected': True,
                'error': f'Prediction failed: {str(predict_error)}'
            }
        
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


@app.route('/test', methods=['GET', 'POST'])
def test():
    """Simple test endpoint to verify app is running."""
    return jsonify({
        'status': 'ok',
        'message': 'App is running',
        'model_initialized': model is not None,
        'face_cascade_initialized': face_cascade is not None,
        'method': request.method
    })


@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    """API endpoint for emotion detection."""
    try:
        # Ensure model is initialized
        if model is None or face_cascade is None:
            print("WARNING: Model not initialized in detect_emotion endpoint. Initializing...")
            print(f"DEBUG: model is None: {model is None}, face_cascade is None: {face_cascade is None}")
            try:
                init_model()
                print(f"DEBUG: After init - model is None: {model is None}, face_cascade is None: {face_cascade is None}")
                if face_cascade is not None:
                    print(f"DEBUG: face_cascade.empty(): {face_cascade.empty()}")
            except Exception as e:
                print(f"ERROR initializing model in endpoint: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Failed to initialize model: {str(e)}', 
                    'face_detected': False,
                    'debug': {
                        'model_initialized': model is not None,
                        'cascade_initialized': face_cascade is not None
                    }
                }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        print(f"DEBUG: Received image data, length: {len(image_data)}")
        print(f"DEBUG: Image data preview (first 100 chars): {image_data[:100]}")
        
        result = detect_emotion_from_image(image_data)
        
        print(f"DEBUG: Detection result - face_detected: {result.get('face_detected')}")
        print(f"DEBUG: Detection result - emotion: {result.get('emotion')}")
        print(f"DEBUG: Detection result - error: {result.get('error')}")
        if 'debug_info' in result:
            print(f"DEBUG: Debug info: {result.get('debug_info')}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR in detect_emotion endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'face_detected': False,
            'debug': {
                'model_initialized': model is not None if 'model' in globals() else 'unknown',
                'cascade_initialized': face_cascade is not None if 'face_cascade' in globals() else 'unknown'
            }
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint to verify model and face cascade are loaded."""
    import cv2
    
    # Try to reinitialize if not loaded
    if model is None or face_cascade is None:
        try:
            print("Health check: Attempting to initialize model...")
            init_model()
        except Exception as e:
            print(f"Health check: Initialization failed: {e}")
    
    status = {
        'status': 'ok',
        'model_loaded': model is not None,
        'face_cascade_loaded': face_cascade is not None,
        'model_path_exists': os.path.exists('emotion_model.h5'),
        'opencv_version': cv2.__version__,
        'opencv_data_path': cv2.data.haarcascades if hasattr(cv2, 'data') else 'N/A',
        'working_directory': os.getcwd(),
        'python_version': sys.version.split()[0]
    }
    
    if face_cascade is not None:
        status['face_cascade_empty'] = face_cascade.empty()
        # Test if cascade can actually detect (with a simple test)
        try:
            test_img = np.zeros((100, 100), dtype=np.uint8)
            test_faces = face_cascade.detectMultiScale(test_img)
            status['cascade_functional'] = True
        except Exception as e:
            status['cascade_functional'] = False
            status['cascade_error'] = str(e)
    else:
        status['face_cascade_empty'] = True
        status['cascade_functional'] = False
    
    # Check if cascade file exists in common locations
    cascade_locations = {
        'default': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None,
        'opencv4_usr': '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        'opencv4_local': '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        'opencv_usr': '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    }
    
    status['cascade_file_locations'] = {}
    for name, path in cascade_locations.items():
        if path:
            status['cascade_file_locations'][name] = {
                'path': path,
                'exists': os.path.exists(path) if path else False
            }
    
    return jsonify(status)


@app.route('/api/test-face-detection', methods=['GET', 'POST'])
def test_face_detection():
    """Test endpoint to debug face detection."""
    try:
        if face_cascade is None:
            return jsonify({
                'error': 'Face cascade not initialized',
                'face_cascade_is_none': True
            }), 500
        
        if face_cascade.empty():
            return jsonify({
                'error': 'Face cascade is empty',
                'face_cascade_empty': True
            }), 500
        
        # If POST, try to detect from provided image
        if request.method == 'POST':
            data = request.get_json()
            if data and 'image' in data:
                image_data = data.get('image')
                result = detect_emotion_from_image(image_data)
                return jsonify({
                    'success': True,
                    'detection_result': result,
                    'cascade_loaded': not face_cascade.empty()
                })
        
        # GET request - create a simple test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Try detection on test image with various parameters
        results = {}
        test_params = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30), 'name': 'strict'},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20), 'name': 'moderate'},
            {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (15, 15), 'name': 'lenient'}
        ]
        
        for params in test_params:
            try:
                faces = face_cascade.detectMultiScale(
                    gray_test,
                    scaleFactor=params['scaleFactor'],
                    minNeighbors=params['minNeighbors'],
                    minSize=params['minSize'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                results[params['name']] = {
                    'faces_detected': len(faces),
                    'params': params
                }
            except Exception as e:
                results[params['name']] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'test_image_shape': test_image.shape,
            'cascade_loaded': not face_cascade.empty(),
            'cascade_empty': face_cascade.empty(),
            'test_results': results,
            'message': 'Face detection system is functional'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


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
# But we catch errors so the app can still start
try:
    print("="*60)
    print("Initializing Sonemote application...")
    print("="*60)
    init_model()
    print("="*60)
    print("Application initialized successfully!")
    print("="*60)
except Exception as e:
    print("="*60)
    print(f"WARNING: Failed to initialize model on startup: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
    print("="*60)
    print("Model will be initialized on first request.")
    print("="*60)

    
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

