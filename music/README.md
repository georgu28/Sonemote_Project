# Music Directory

This directory contains emotion-specific music folders. The Sonemote application will automatically play music from the appropriate folder based on the detected emotion.

## Directory Structure

```
music/
├── angry/      - Aggressive, intense music (rock, metal, electronic)
├── disgust/    - Dark, ambient music
├── fear/       - Tense, suspenseful music
├── happy/      - Upbeat, cheerful music (pop, dance, upbeat)
├── sad/        - Melancholic, emotional music (ballads, slow)
├── surprise/   - Energetic, exciting music
└── neutral/    - Calm, ambient music
```

## Supported Audio Formats

- MP3 (`.mp3`)
- WAV (`.wav`)
- OGG (`.ogg`)
- M4A (`.m4a`)
- FLAC (`.flac`)

## Adding Music Files

1. Place your music files in the appropriate emotion folder
2. The application will automatically detect and play them
3. Files are shuffled randomly for variety
4. You can have multiple tracks per emotion category

## Example

If you detect a "Happy" emotion, the application will:
1. Select a random track from `music/happy/`
2. Play it with a smooth transition
3. Switch to a different track if the emotion changes

## Notes

- The application uses a 3-second delay before switching music to avoid rapid changes
- Music transitions smoothly with fade in/out effects
- You can control playback with keyboard shortcuts (see main README)

## Getting Sample Music

You can:
- Use royalty-free music from sites like:
  - [Free Music Archive](https://freemusicarchive.org/)
  - [Incompetech](https://incompetech.com/music/royalty-free/)
  - [Bensound](https://www.bensound.com/)
- Create your own playlists
- Use music you have rights to use

Make sure to organize tracks by emotion for the best experience!

