// Sonemote Web App JavaScript

class SonemoteApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.currentEmotion = null;
        this.currentEmotionCategory = 'Happy';
        this.audioPlayer = document.getElementById('audio-player');
        
        // Emotion change delay tracking
        this.pendingEmotion = null;
        this.pendingEmotionTimeout = null;
        this.emotionChangeDelay = 1000; // 1 second delay
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSongs(this.currentEmotionCategory);
        // Auto-start camera on page load (with error handling)
        this.startCamera().catch(error => {
            console.log('Camera auto-start failed (user may need to grant permission):', error);
            // Don't show alert for auto-start failure - let user start manually
        });
    }

    setupEventListeners() {
        // Camera controls
        document.getElementById('start-camera').addEventListener('click', () => this.startCamera());
        document.getElementById('stop-camera').addEventListener('click', () => this.stopCamera());
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const emotion = e.target.dataset.emotion;
                this.switchTab(emotion);
            });
        });
        
        // File upload
        document.getElementById('upload-btn').addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
        
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.uploadFile(e.target.files[0]);
        });
        
        // YouTube search
        document.getElementById('search-btn').addEventListener('click', () => {
            this.searchYouTube();
        });
        
        document.getElementById('youtube-search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchYouTube();
            }
        });
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.video.play();
            
            // Start emotion detection
            this.startDetection();
            
            // Update UI
            document.getElementById('start-camera').disabled = true;
            document.getElementById('stop-camera').disabled = false;
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            // Only show alert if user explicitly clicked start button
            if (error.name !== 'NotAllowedError') {
                alert('Could not access camera. Please check permissions and try clicking "Start Camera" button.');
            }
            throw error; // Re-throw so caller can handle it
        }
    }

    stopCamera() {
        // Stop video stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Stop detection
        this.stopDetection();
        
        // Clear video
        this.video.srcObject = null;
        
        // Update UI
        document.getElementById('start-camera').disabled = false;
        document.getElementById('stop-camera').disabled = true;
        this.updateEmotionDisplay('No Face Detected', 0);
    }

    startDetection() {
        if (this.isDetecting) return;
        
        this.isDetecting = true;
        
        // Detect emotion every 500ms
        this.detectionInterval = setInterval(() => {
            this.detectEmotion();
        }, 500);
    }

    stopDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        this.isDetecting = false;
        
        // Clear any pending emotion change
        if (this.pendingEmotionTimeout) {
            clearTimeout(this.pendingEmotionTimeout);
            this.pendingEmotionTimeout = null;
            this.pendingEmotion = null;
        }
    }

    detectEmotion() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;
        
        // Set canvas size
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw video frame to canvas (mirrored)
        this.ctx.save();
        this.ctx.scale(-1, 1);
        this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
        this.ctx.restore();
        
        // Convert to base64
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to backend
        fetch('/api/detect-emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Detection error:', data.error);
                return;
            }
            
            if (data.face_detected && data.emotion) {
                this.updateEmotionDisplay(data.emotion, data.confidence);
                
                // Handle emotion change with 1 second delay
                if (data.emotion !== this.currentEmotion) {
                    // If there's a different pending emotion, cancel it
                    if (this.pendingEmotion !== data.emotion) {
                        if (this.pendingEmotionTimeout) {
                            clearTimeout(this.pendingEmotionTimeout);
                            this.pendingEmotionTimeout = null;
                        }
                        
                        // Set new pending emotion
                        this.pendingEmotion = data.emotion;
                        
                        // Wait 1 second before actually changing
                        this.pendingEmotionTimeout = setTimeout(() => {
                            // Only change if the pending emotion is still valid and different from current
                            if (this.pendingEmotion && this.pendingEmotion !== this.currentEmotion) {
                                this.currentEmotion = this.pendingEmotion;
                                this.onEmotionChange(this.pendingEmotion);
                            }
                            this.pendingEmotion = null;
                            this.pendingEmotionTimeout = null;
                        }, this.emotionChangeDelay);
                    }
                } else {
                    // Same emotion detected - cancel any pending change
                    if (this.pendingEmotionTimeout) {
                        clearTimeout(this.pendingEmotionTimeout);
                        this.pendingEmotionTimeout = null;
                        this.pendingEmotion = null;
                    }
                }
            } else {
                this.updateEmotionDisplay('No Face Detected', 0);
                // Cancel pending emotion change if no face detected
                if (this.pendingEmotionTimeout) {
                    clearTimeout(this.pendingEmotionTimeout);
                    this.pendingEmotionTimeout = null;
                    this.pendingEmotion = null;
                }
            }
        })
        .catch(error => {
            console.error('Error detecting emotion:', error);
        });
    }

    updateEmotionDisplay(emotion, confidence) {
        const emotionLabel = document.querySelector('.emotion-label');
        const emotionConfidence = document.querySelector('.emotion-confidence');
        
        emotionLabel.textContent = emotion || 'No Face Detected';
        emotionLabel.className = `emotion-label emotion-${emotion ? emotion.toLowerCase() : 'neutral'}`;
        
        if (confidence > 0) {
            emotionConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        } else {
            emotionConfidence.textContent = '';
        }
    }

    onEmotionChange(emotion) {
        // Update current emotion display
        document.getElementById('current-emotion').textContent = emotion;
        
        // Load and play music for this emotion
        this.loadAndPlayMusic(emotion);
    }

    async loadAndPlayMusic(emotion) {
        try {
            const response = await fetch(`/api/music/${emotion}`);
            const data = await response.json();
            
            if (data.songs && data.songs.length > 0) {
                // Play first song (or random)
                const randomSong = data.songs[Math.floor(Math.random() * data.songs.length)];
                this.playSong(randomSong.path, randomSong.name);
            } else {
                document.getElementById('current-track').textContent = `No songs available for ${emotion}`;
                this.audioPlayer.src = '';
            }
        } catch (error) {
            console.error('Error loading music:', error);
        }
    }

    playSong(path, name) {
        this.audioPlayer.src = path;
        document.getElementById('current-track').textContent = name;
        this.audioPlayer.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }

    switchTab(emotion) {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.emotion === emotion) {
                btn.classList.add('active');
            }
        });
        
        // Update category name
        document.getElementById('current-category').textContent = emotion;
        
        // Load songs for this emotion
        this.currentEmotionCategory = emotion;
        this.loadSongs(emotion);
    }

    async loadSongs(emotion) {
        const container = document.getElementById('songs-container');
        container.innerHTML = '<p class="loading">Loading songs...</p>';
        
        try {
            const response = await fetch(`/api/music/${emotion}`);
            const data = await response.json();
            
            if (data.songs && data.songs.length > 0) {
                container.innerHTML = data.songs.map(song => `
                    <div class="song-item">
                        <span class="song-name">${song.name}</span>
                        <div class="song-actions">
                            <button class="btn-icon" onclick="app.playSong('${song.path}', '${song.name}')" title="Play">
                                ▶️
                            </button>
                            <button class="btn-icon danger" onclick="app.deleteSong('${emotion}', '${song.filename}')" title="Delete">
                                🗑️
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<p class="empty-state">No songs in this category. Upload some music!</p>';
            }
        } catch (error) {
            console.error('Error loading songs:', error);
            container.innerHTML = '<p class="empty-state">Error loading songs. Please try again.</p>';
        }
    }

    async uploadFile(file) {
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`/api/music/${this.currentEmotionCategory}`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert(`File "${data.filename}" added successfully!`);
                this.loadSongs(this.currentEmotionCategory);
                // Clear file input
                document.getElementById('file-input').value = '';
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file. Please try again.');
        }
    }

    async deleteSong(emotion, filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/music/${emotion}/${filename}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.loadSongs(emotion);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error deleting song:', error);
            alert('Error deleting song. Please try again.');
        }
    }
    
    async searchYouTube() {
        const query = document.getElementById('youtube-search-input').value.trim();
        if (!query) {
            alert('Please enter a search query');
            return;
        }
        
        const searchBtn = document.getElementById('search-btn');
        const resultsContainer = document.getElementById('search-results');
        const resultsList = document.getElementById('search-results-list');
        
        // Show loading state
        searchBtn.disabled = true;
        searchBtn.textContent = 'Searching...';
        resultsContainer.style.display = 'block';
        resultsList.innerHTML = '<p class="loading">Searching YouTube...</p>';
        
        try {
            const response = await fetch(`/api/youtube/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            if (data.error) {
                resultsList.innerHTML = `<p class="empty-state">Error: ${data.error}</p>`;
                return;
            }
            
            if (data.results && data.results.length > 0) {
                const currentCategory = this.currentEmotionCategory;
                resultsList.innerHTML = data.results.map((video, index) => `
                    <div class="search-result-item">
                        <div class="result-info">
                            <div class="result-title">${this.escapeHtml(video.title)}</div>
                            <div class="result-duration">${this.formatDuration(video.duration)}</div>
                        </div>
                        <button 
                            class="btn btn-primary btn-sm" 
                            onclick="app.downloadYouTube('${video.url.replace(/'/g, "\\'")}', '${currentCategory}')"
                            title="Add to ${currentCategory}"
                        >
                            ➕ Add
                        </button>
                    </div>
                `).join('');
            } else {
                resultsList.innerHTML = '<p class="empty-state">No results found. Try a different search.</p>';
            }
        } catch (error) {
            console.error('Error searching YouTube:', error);
            resultsList.innerHTML = '<p class="empty-state">Error searching YouTube. Please try again.</p>';
        } finally {
            searchBtn.disabled = false;
            searchBtn.textContent = 'Search';
        }
    }
    
    async downloadYouTube(url, emotion) {
        // Show loading message
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'loading-overlay';
        loadingMsg.innerHTML = `
            <div class="loading-content">
                <p>Downloading audio from YouTube...</p>
                <p class="loading-hint">This may take a minute</p>
            </div>
        `;
        document.body.appendChild(loadingMsg);
        
        try {
            const response = await fetch('/api/youtube/download', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url, emotion })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert(`"${data.title}" has been added to ${emotion} category!`);
                // Clear search results
                document.getElementById('search-results').style.display = 'none';
                document.getElementById('youtube-search-input').value = '';
                // Reload songs
                this.loadSongs(emotion);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error downloading YouTube video:', error);
            alert('Error downloading audio. Please try again.');
        } finally {
            document.body.removeChild(loadingMsg);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatDuration(seconds) {
        if (!seconds) return 'Unknown';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new SonemoteApp();
});

