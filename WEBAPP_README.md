# FaceTune Web Application

A beautiful web interface for the FaceTune emotion-based music player. This Flask webapp allows users to interact with the emotion detection model through their browser, manage music playlists, and enjoy real-time emotion-based music playback.

## Features

-  **Real-time Camera Feed**: Access your webcam directly in the browser (auto-starts on page load)
-  **Emotion Detection**: Real-time emotion recognition from facial expressions
-  **Music Management**: Add songs via YouTube search or file upload, remove, and organize songs for each emotion category
-  **Automatic Playback**: Music automatically plays based on detected emotions
-  **YouTube Integration**: Search and download music directly from YouTube
-  **Responsive Design**: Beautiful, modern UI that works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: For YouTube downloads to work, you also need **FFmpeg** installed:

#### Windows Installation (Step-by-Step):

1. **Download FFmpeg Windows Build**
   - Go to https://www.gyan.dev/ffmpeg/builds/ (recommended) or https://ffmpeg.org/download.html
   - **Important**: Download the **Windows executable build** (not the source code)
   - Choose "ffmpeg-release-essentials.zip" or "ffmpeg-release-full.zip"
   - The file should be named something like `ffmpeg-release-essentials.zip`

2. **Extract the ZIP File**
   - Extract the downloaded ZIP file to a location like `C:\ffmpeg` or `C:\Program Files\ffmpeg`
   - After extraction, you should see folders like `bin`, `doc`, `presets`, etc.
   - The important folder is `bin` which contains `ffmpeg.exe` and `ffprobe.exe`

3. **Add FFmpeg to PATH**
   - Press `Windows Key + X` and select "System"
   - Click "Advanced system settings" on the left
   - Click "Environment Variables" button at the bottom
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add the path to the `bin` folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs to save

4. **Verify Installation**
   - Open a **new** Command Prompt or PowerShell window (important: must be new to pick up PATH changes)
   - Type: `ffmpeg -version`
   - Type: `ffprobe -version`
   - If both commands show version information, FFmpeg is installed correctly!
   - If you get "not recognized" error, check that you added the correct `bin` folder path and restart your terminal

**Alternative**: If you don't want to modify PATH, you can set the `FFMPEG_LOCATION` environment variable:
   - Set it to the full path to `ffmpeg.exe` (e.g., `C:\ffmpeg\bin\ffmpeg.exe`)

#### Mac Installation:
```bash
brew install ffmpeg
```

#### Linux Installation:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# Fedora/CentOS
sudo yum install ffmpeg
```

### 2. Run the Web Application

```bash
python app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

### Starting the Camera

1. The camera **auto-starts** when you load the page (if permissions are granted)
2. If auto-start fails, click the **"Start Camera"** button manually
3. Allow camera permissions when prompted by your browser
4. Position yourself in front of the camera
5. The app will detect your emotions in real-time

### Adding Music from YouTube

1. Select an emotion category tab (Happy, Sad, Angry, etc.)
2. In the **search bar**, type the name of a song or artist
3. Click **"Search"** to find YouTube videos
4. Click **"➕ Add"** next to any search result to download and add it to that emotion's playlist
5. The audio will be downloaded and converted to MP3 format

**Note**: YouTube downloads require **FFmpeg** to be installed on your system for audio conversion.

### Adding Music from File Upload

1. Select an emotion category tab (Happy, Sad, Angry, etc.)
2. Click **"Upload Audio File"**
3. Choose an audio file (MP3, WAV, OGG, M4A, or FLAC)
4. The file will be added to that emotion's playlist

### Removing Music

1. Select an emotion category
2. Find the song in the playlist
3. Click the 🗑️ delete button next to the song

### Playing Music

- Music automatically plays when an emotion is detected
- Click the ▶️ play button next to any song to play it manually
- Use the audio player controls to pause, adjust volume, etc.

## Emotion Categories

- **Happy** 🟡 - Upbeat, cheerful music
- **Sad** 🔵 - Melancholic, emotional music
- **Angry** 🔴 - Aggressive, intense music
- **Fear** 🟣 - Tense, suspenseful music
- **Surprise** 🟢 - Energetic, exciting music
- **Disgust** 🟠 - Dark, ambient music
- **Neutral** ⚪ - Calm, ambient music

## Technical Details

### Backend (Flask)

- **Emotion Detection API**: `/api/detect-emotion` - Receives base64 encoded images and returns emotion predictions
- **Music Management API**: 
  - `GET /api/music/<emotion>` - Get list of songs for an emotion
  - `POST /api/music/<emotion>` - Upload a song file
  - `DELETE /api/music/<emotion>/<filename>` - Delete a song
  - `GET /api/music/file/<emotion>/<filename>` - Stream a song file
- **YouTube API**:
  - `GET /api/youtube/search?q=<query>` - Search YouTube for videos
  - `POST /api/youtube/download` - Download audio from YouTube URL

### Frontend

- **Camera Access**: Uses `getUserMedia` API for webcam access
- **Image Capture**: Captures frames from video stream and sends to backend for processing
- **Audio Playback**: Uses HTML5 `<audio>` element for music playback
- **Real-time Updates**: Emotion detection runs every 500ms

## Troubleshooting

### Camera Not Working

- Ensure you've granted camera permissions in your browser
- Check that no other application is using the camera
- Try refreshing the page and allowing permissions again

### Model Not Loading

- Ensure `emotion_model.h5` exists in the project root
- If the model doesn't exist, the app will use an untrained model structure (won't work properly)
- Train the model first: `python src/train.py`

### Music Not Playing

- Check that audio files are in supported formats (MP3, WAV, OGG, M4A, FLAC)
- Ensure music files are uploaded to the correct emotion folders
- Check browser console for audio playback errors

### YouTube Download Not Working

- **FFmpeg not installed**: Install FFmpeg (see installation instructions above)
- **Download fails**: Check your internet connection and try again
- **File not found after download**: Check the `music/<emotion>/` folder to see if the file was saved with a different name
- **Search not working**: Check server logs for errors. YouTube search depends on yt-dlp working correctly.

### Songs Not Appearing

- Refresh the page after uploading a file
- Check that files are being saved to the `music/<emotion>/` directories
- Verify file permissions allow reading/writing

## Development

### Project Structure

```
FaceTune_Project/
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Stylesheet
│   └── js/
│       └── app.js        # Frontend JavaScript
├── music/                # Music files organized by emotion
│   ├── happy/
│   ├── sad/
│   └── ...
└── emotion_model.h5      # Trained emotion detection model
```

### Adding New Features

The app is structured to be easily extensible:

- **New API endpoints**: Add routes to `app.py`
- **Frontend changes**: Modify `templates/index.html` and `static/js/app.js`
- **Styling**: Update `static/css/style.css`

## Comparison: Flask vs Streamlit vs React

### Why Flask? (Current Implementation)

✅ **Pros:**
- Full control over frontend and backend
- Better camera handling with getUserMedia API
- More flexible for custom UI/UX
- Can integrate with existing Python code easily
- Good performance for real-time applications

❌ **Cons:**
- More setup required
- Need to write HTML/CSS/JS manually

### Alternative: Streamlit

✅ **Pros:**
- Very quick to prototype
- No HTML/CSS/JS needed
- Great for data science apps

❌ **Cons:**
- Limited camera handling (needs workarounds)
- Less control over UI customization
- Harder to implement real-time updates
- Less suitable for music playback controls

### Alternative: React + Flask

✅ **Pros:**
- Most flexible and modern
- Best for complex interactive UIs
- Can use React frameworks (Next.js, etc.)

❌ **Cons:**
- Most complex setup
- Requires knowledge of React/Node.js
- More dependencies to manage
- Overkill for this use case

**Recommendation**: Flask (current implementation) is the sweet spot - easier than React but more flexible than Streamlit, perfect for this application!

## License

This project is for educational purposes.

