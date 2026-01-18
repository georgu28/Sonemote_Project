# Sonemote - Emotion-Based Music Player

Real-time emotion detection from webcam that automatically plays music matching your emotions.

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Sonemote_Project

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The project uses the FER2013 dataset. You have two options:

**Option A: Use folder structure (recommended)**
- The `data/train/` and `data/test/` folders should already contain images organized by emotion
- If not, download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract

**Option B: Use CSV file**
- Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Place it in the `data/` directory

### 3. Train the Model

```bash
python src/train.py
```

This will:
- Load training data (28,709 samples)
- Train for up to 100 epochs (with early stopping)
- Save the best model as `emotion_model.h5`
- Take approximately 1-3 hours depending on your hardware

**Note:** If you already have a trained model (`emotion_model.h5`), skip this step.

### 4. Add Music (Optional)

Add music files to emotion-specific folders:

```
music/
├── angry/      # Aggressive, intense music
├── disgust/    # Dark, ambient music
├── fear/       # Tense, suspenseful music
├── happy/      # Upbeat, cheerful music
├── sad/        # Melancholic, emotional music
├── surprise/   # Energetic, exciting music
└── neutral/    # Calm, ambient music
```

Supported formats: MP3, WAV, OGG, M4A, FLAC

### 5. Run the Application

pyth**Option A: Web Application (Flask)**

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

**Option B: Command Line Interface**

```bash
python src/infer.py
```

**Controls (CLI):**
- `q` - Quit
- `p` - Pause/Resume music
- `+` - Increase volume
- `-` - Decrease volume
- `n` - Next track
- `s` - Stop music

## Test Mode (No Training Required)

Test the application without training a model:

```bash
python src/infer.py --test-mode
```

This uses mock predictions so you can test camera, face detection, and music playback immediately.

## Requirements

- Python 3.8+
- Webcam
- Trained model (`emotion_model.h5`) - created in step 3
- Music files (optional) - for music playback feature

## Troubleshooting

**Model not found?**
- Train the model: `python src/train.py`
- Or use test mode: `python src/infer.py --test-mode`

**Camera not working?**
- Check camera permissions
- Try different camera: `python src/infer.py --camera 1`

**Music not playing?**
- Add music files to `music/[emotion]/` folders
- Check that pygame is installed: `pip install pygame`

## Project Structure

```
Sonemote_Project/
├── data/              # Training/test images
├── src/               # Source code
│   ├── train.py      # Train the model
│   ├── infer.py      # Run emotion detection
│   └── ...
├── music/            # Emotion-specific music folders
├── emotion_model.h5  # Trained model (created after training)
└── requirements.txt  # Python dependencies
```

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade to detect faces in webcam feed
2. **Emotion Recognition**: CNN model classifies emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
3. **Music Playback**: Automatically plays music from the matching emotion folder
4. **Smart Switching**: Only switches music when emotion is stable (5 consecutive frames)

## Model Details

- **Architecture**: VGG-style CNN with 2.6M parameters
- **Input**: 48x48 grayscale images
- **Output**: 7 emotion classes
- **Expected Accuracy**: ~60-65% on FER2013 test set

## License

This project is for educational purposes. Please refer to the FER2013 dataset license for data usage terms.
