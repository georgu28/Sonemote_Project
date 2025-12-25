"""
Real-time emotion detection using webcam with music playback.
Uses OpenCV for face detection and Keras model for emotion prediction.
Plays music based on detected emotions.
"""

import cv2
import numpy as np
from tensorflow import keras
import os
import sys
import random

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_player import EmotionMusicPlayer
from models import create_emotion_model


def preprocess_face(face_roi, target_size=(48, 48)):
    """
    Preprocess face ROI for model prediction.
    
    Args:
        face_roi: Face region of interest (BGR image)
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed face ready for model input
    """
    # Convert BGR to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size (48x48)
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Reshape to (1, 48, 48, 1) for model input
    processed = normalized.reshape(1, target_size[1], target_size[0], 1)
    
    return processed


def create_mock_predictions():
    """
    Create a mock prediction function for testing without a trained model.
    Returns random emotion predictions with varying confidence.
    """
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def mock_predict(processed_face):
        # Generate random predictions with bias
        weights = np.array([0.08, 0.23, 0.26, 0.02, 0.21, 0.08, 0.12])
        predictions = np.random.dirichlet(weights * 10)  # Scale for more confidence
        return predictions.reshape(1, -1)
    
    return mock_predict


def apply_emotion_weights(predictions):
    """
    Apply weighting to emotion predictions to favor harder-to-detect emotions.
    Increases weights for: Sad, Fear, Disgust
    Decreases weight for: Neutral
    
    Args:
        predictions: Raw prediction array (1, 7) or (7,)
        
    Returns:
        Weighted predictions (same shape as input)
    """
    # Emotion order: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Weight multipliers: increase Disgust (index 1), decrease Neutral (index 6)
    weight_multipliers = np.array([0.7, 3.0, 1.8, 0.5, 2.0, 1.0, 0.3])  
    
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


def realtime_detect(model_path='emotion_model.h5', 
                    cascade_path=None,
                    camera_index=0,
                    music_dir='music',
                    enable_music=True,
                    min_confidence=0.30,  # Lower default for more sensitivity (~25-30% confidence)
                    test_mode=False):
    """
    Real-time emotion detection from webcam with music playback.
    
    Args:
        model_path: Path to the trained model file
        cascade_path: Path to Haar cascade file for face detection
                     (if None, uses default OpenCV cascade)
        camera_index: Camera device index (0 for default webcam)
        music_dir: Directory containing emotion-specific music folders
        enable_music: Whether to enable music playback
        min_confidence: Minimum confidence threshold for emotion detection (default: 0.25, allows ~25-30% confidence)
        test_mode: If True, use mock predictions instead of trained model
    """
    print("="*60)
    print("REAL-TIME EMOTION DETECTION")
    if test_mode:
        print("⚠️  TEST MODE - Using mock predictions (no trained model)")
    print("="*60)
    
    # Load model or use test mode
    model = None
    use_mock = False
    mock_predict_fn = None
    
    if test_mode or not os.path.exists(model_path):
        if test_mode:
            print("\n⚠️  Test mode enabled - using mock predictions")
        else:
            print(f"\n⚠️  Model file not found: {model_path}")
            print("Switching to TEST MODE with mock predictions...")
            print("(To train a model, run: python src/train.py)")
        
        # Create untrained model for structure, but use mock predictions
        print("Creating model architecture (for structure only)...")
        model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
        mock_predict_fn = create_mock_predictions()
        use_mock = True
    else:
        print(f"\nLoading trained model from {model_path}...")
        model = keras.models.load_model(model_path)
        use_mock = False
    
    # Load face cascade classifier
    if cascade_path is None:
        # Try to use OpenCV's default cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            f"Face cascade file not found: {cascade_path}\n"
            "Please download haarcascade_frontalface_default.xml or specify cascade_path"
        )
    
    print(f"Loading face cascade from {cascade_path}...")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Initialize music player
    music_player = None
    if enable_music:
        try:
            music_player = EmotionMusicPlayer(music_dir=music_dir, volume=0.6, transition_delay=3.0)
            print("\nMusic player initialized successfully!")
        except Exception as e:
            print(f"\nWarning: Could not initialize music player: {e}")
            print("Continuing without music playback...")
            enable_music = False
    
    # Emotion labels mapping (FER2013 uses 0-6)
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Colors for bounding boxes (BGR format)
    emotion_colors = {
        'Angry': (0, 0, 255),      # Red
        'Disgust': (0, 128, 255),   # Orange
        'Fear': (128, 0, 128),      # Purple
        'Happy': (0, 255, 255),     # Yellow
        'Sad': (255, 0, 0),         # Blue
        'Surprise': (0, 255, 0),    # Green
        'Neutral': (128, 128, 128)  # Gray
    }
    
    # Emotion stability tracking (to avoid rapid switching)
    emotion_stability = {}  # Track how long each emotion has been detected
    stable_emotion = None
    stable_confidence = 0.0
    stability_threshold = 5  # Number of consecutive frames to consider emotion "stable" (reduced for sensitivity)
    # Use a more lenient confidence threshold for stability tracking
    stability_confidence_threshold = 0.30  # Lower threshold for consistent emotions (~25% is acceptable)
    
    # Open webcam
    print(f"\nOpening webcam (camera index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam. Please check if camera is connected.")
    
    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Emotion detection started!")
    if use_mock:
        print("⚠️  NOTE: Using random predictions for testing")
        print("   Face detection and music playback are fully functional")
    print("\nControls:")
    print("  'q' - Quit")
    if enable_music:
        print("  'p' - Pause/Resume music")
        print("  '+' - Increase volume")
        print("  '-' - Decrease volume")
        print("  'n' - Next track")
        print("  's' - Stop music")
    print("="*60 + "\n")
    
    try:
        while True:
            # Check if current song ended and play next random song
            if enable_music and music_player:
                music_player.check_music_end()
            
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from webcam.")
                break
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Reset stability if no faces detected
            if len(faces) == 0:
                emotion_stability.clear()
                stable_emotion = None
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess face for model
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion (use mock if in test mode)
                if use_mock:
                    predictions = mock_predict_fn(processed_face)
                else:
                    predictions = model.predict(processed_face, verbose=0)
                
                # Apply weighting to favor harder-to-detect emotions (Sad, Fear, Disgust)
                predictions = apply_emotion_weights(predictions)
                
                emotion_idx = np.argmax(predictions[0])
                confidence = predictions[0][emotion_idx]
                
                # Get emotion label and color
                emotion_label = emotion_labels[emotion_idx]
                color = emotion_colors.get(emotion_label, (255, 255, 255))
                
                # Track emotion stability with more lenient threshold
                # Use lower threshold for stability tracking (allows ~25-30% confidence)
                stability_check_confidence = max(min_confidence, stability_confidence_threshold)
                
                if confidence >= stability_check_confidence:
                    # Update stability counter for current emotion
                    if emotion_label in emotion_stability:
                        emotion_stability[emotion_label] += 1
                    else:
                        emotion_stability[emotion_label] = 1
                    
                    # Reset counters for other emotions (but don't clear if they're close)
                    for other_emotion in emotion_labels.values():
                        if other_emotion != emotion_label:
                            emotion_stability[other_emotion] = 0
                    
                    # Check if emotion is stable enough to switch music
                    if emotion_stability[emotion_label] >= stability_threshold:
                        # Only update if emotion actually changed
                        if emotion_label != stable_emotion:
                            stable_emotion = emotion_label
                            stable_confidence = confidence
                            
                            # Update music player - immediate switch for stable emotion
                            if enable_music and music_player:
                                music_player.switch_emotion_immediate(emotion_label)
                                print(f"✓ Music switched to: {emotion_label} (confidence: {confidence:.2f}, stable for {emotion_stability[emotion_label]} frames)")
                else:
                    # Low confidence - only reset if it's really low (below stability threshold)
                    # This allows for slight fluctuations without resetting progress
                    if confidence < stability_confidence_threshold:
                        # Only reset the current emotion's counter, not all
                        if emotion_label in emotion_stability:
                            emotion_stability[emotion_label] = max(0, emotion_stability[emotion_label] - 1)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare text with emotion and confidence
                # Show stable emotion if different from current detection
                if stable_emotion and stable_emotion != emotion_label:
                    text = f"{emotion_label}: {confidence:.2f} → {stable_emotion}"
                else:
                    text = f"{emotion_label}: {confidence:.2f}"
                
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 10),
                    (x + text_width, y),
                    color,
                    -1
                )
                
                # Draw emotion label and confidence
                cv2.putText(
                    frame,
                    text,
                    (x, y - 5),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    cv2.LINE_AA
                )
            
            # Display music status if enabled
            if enable_music and music_player:
                status = music_player.get_status()
                music_text = f"Music: {status['track']}"
                if status['paused']:
                    music_text += " (PAUSED)"
                elif not status['playing']:
                    music_text += " (STOPPED)"
                
                # Draw music status at bottom of frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    music_text, font, font_scale, thickness
                )
                
                # Background for music text
                cv2.rectangle(
                    frame,
                    (10, frame.shape[0] - text_height - 15),
                    (10 + text_width + 10, frame.shape[0] - 5),
                    (0, 0, 0),
                    -1
                )
                
                # Music status text
                cv2.putText(
                    frame,
                    music_text,
                    (15, frame.shape[0] - 10),
                    font,
                    font_scale,
                    (0, 255, 0),  # Green text
                    thickness,
                    cv2.LINE_AA
                )
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif enable_music and music_player:
                if key == ord('p'):
                    if music_player.is_paused:
                        music_player.play()
                    else:
                        music_player.pause()
                elif key == ord('+') or key == ord('='):
                    music_player.increase_volume(0.1)
                elif key == ord('-') or key == ord('_'):
                    music_player.decrease_volume(0.1)
                elif key == ord('n'):
                    music_player.next_track()
                elif key == ord('s'):
                    music_player.stop()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        # Clean up
        print("\nReleasing camera...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up music player
        if music_player:
            print("Stopping music player...")
            music_player.cleanup()
        
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time emotion detection with music playback')
    parser.add_argument('--model', type=str, default='emotion_model.h5',
                        help='Path to trained model file (default: emotion_model.h5)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--music-dir', type=str, default='music',
                        help='Directory containing emotion-specific music folders (default: music)')
    parser.add_argument('--no-music', action='store_true',
                        help='Disable music playback')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use test mode with mock predictions (no trained model needed)')
    parser.add_argument('--min-confidence', type=float, default=0.30,
                        help='Minimum confidence threshold (default: 0.25, allows ~25-30% confidence)')
    
    args = parser.parse_args()
    
    # Run real-time detection with music
    try:
        realtime_detect(
            model_path=args.model,
            camera_index=args.camera,
            music_dir=args.music_dir,
            enable_music=not args.no_music,
            min_confidence=args.min_confidence,
            test_mode=args.test_mode
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. A webcam connected to your computer")
        print("  2. OpenCV installed and configured correctly")
        if not args.test_mode:
            print("  3. Trained the model using train.py (or use --test-mode)")
        print("  4. Music files in the 'music/' directory (optional)")
        print("\nTo test without a trained model, use:")
        print("  python src/infer.py --test-mode")

