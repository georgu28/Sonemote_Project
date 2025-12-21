# Sonemote Web App - Deployment Guide

This guide covers how to deploy the Sonemote web application to various hosting platforms so it's accessible to everyone via a public URL.

## Prerequisites

Before deploying, ensure you have:
- ✅ Trained model file (`emotion_model.h5`) in the project root
- ✅ Python 3.8+ installed
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ FFmpeg installed (for YouTube downloads feature)
  - **Windows**: See detailed installation steps in "Troubleshooting FFmpeg Issues" section below
  - **Mac**: `brew install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg`

## Deployment Options

### Option 1: Render (Recommended for Easy Deployment)

[Render](https://render.com) is a great platform for Flask apps with free tier available.

#### Steps:

1. **Create a Render Account**
   - Go to https://render.com and sign up

2. **Prepare Your Repository**
   - Push your code to GitHub/GitLab/Bitbucket
   - Ensure `requirements.txt` is in the root directory

3. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your repository
   - Configure:
     - **Name**: `sonemote-app` (or your choice)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py` (or `gunicorn app:app` for production)
     - **Instance Type**: Free tier is fine for testing

4. **Set Environment Variables**
   - In Render dashboard, go to "Environment" tab
   - Add if needed:
     - `FFMPEG_LOCATION`: `/usr/bin/ffmpeg` (if FFmpeg is installed)
     - `PORT`: Render sets this automatically

5. **Install FFmpeg on Render**
   - Render doesn't include FFmpeg by default
   - Add a `render.yaml` file to your repo:
   ```yaml
   services:
     - type: web
       name: sonemote-app
       env: python
       buildCommand: |
         apt-get update && apt-get install -y ffmpeg
         pip install -r requirements.txt
       startCommand: python app.py
   ```
   - Or use a Dockerfile (see Option 2)

6. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your app
   - Your app will be available at: `https://sonemote-app.onrender.com`

**Note**: Free tier on Render spins down after inactivity. Consider paid tier for production.

---

### Option 2: Railway

[Railway](https://railway.app) offers easy deployment with built-in FFmpeg support.

#### Steps:

1. **Create Railway Account**
   - Go to https://railway.app and sign up

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Build**
   - Railway auto-detects Python projects
   - Add `Procfile` (optional):
     ```
     web: python app.py
     ```

4. **Set Environment Variables**
   - Go to "Variables" tab
   - Add: `PORT=5000` (Railway sets this automatically, but good to have)

5. **Install FFmpeg**
   - Railway supports FFmpeg via `nixpacks` or Dockerfile
   - Create `nixpacks.toml`:
     ```toml
     [phases.setup]
     nixPkgs = ["python39", "ffmpeg"]
     
     [phases.install]
     cmds = ["pip install -r requirements.txt"]
     
     [start]
     cmd = "python app.py"
     ```

6. **Deploy**
   - Railway will automatically deploy
   - Your app URL will be: `https://your-app-name.up.railway.app`

---

### Option 3: Heroku

[Heroku](https://heroku.com) is a popular platform (note: free tier discontinued, but paid tiers available).

#### Steps:

1. **Install Heroku CLI**
   ```bash
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   
   # Mac/Linux
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create sonemote-app
   ```

3. **Create Procfile**
   Create `Procfile` in project root:
   ```
   web: gunicorn app:app
   ```

4. **Add Buildpack for FFmpeg**
   ```bash
   heroku buildpacks:add --index 1 heroku-community/apt
   heroku buildpacks:add --index 2 heroku/python
   ```

5. **Create Aptfile**
   Create `Aptfile` in project root:
   ```
   ffmpeg
   ```

6. **Update requirements.txt**
   Add `gunicorn`:
   ```
   gunicorn>=21.2.0
   ```

7. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

8. **Set Config Vars**
   ```bash
   heroku config:set FFMPEG_LOCATION=/usr/bin/ffmpeg
   ```

---

### Option 4: DigitalOcean App Platform

[DigitalOcean App Platform](https://www.digitalocean.com/products/app-platform) offers straightforward deployment.

#### Steps:

1. **Create DigitalOcean Account**
   - Go to https://www.digitalocean.com

2. **Create App**
   - Go to "Apps" → "Create App"
   - Connect your repository

3. **Configure App**
   - **Type**: Web Service
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `gunicorn app:app --bind 0.0.0.0:8080`
   - **Environment Variables**: Add `PORT=8080`

4. **Add FFmpeg**
   - Use Dockerfile (see below) or add to build command:
   ```
   apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt
   ```

5. **Deploy**
   - Click "Create Resources"
   - Your app will be available at: `https://your-app-name.ondigitalocean.app`

---

### Option 5: Docker + Any Platform (Most Flexible)

Using Docker allows deployment to any platform (AWS, Google Cloud, Azure, etc.).

#### Create Dockerfile:

```dockerfile
FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variable for FFmpeg
ENV FFMPEG_LOCATION=/usr/bin/ffmpeg

# Run the application
CMD ["python", "app.py"]
```

#### Create .dockerignore:

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
myenv/
venv/
env/
.git
.gitignore
*.md
.DS_Store
```

#### Build and Run Locally:

```bash
docker build -t sonemote-app .
docker run -p 5000:5000 sonemote-app
```

#### Deploy to Cloud:

- **AWS**: Use AWS ECS, Elastic Beanstalk, or EC2
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Use Azure Container Instances or App Service
- **Fly.io**: `flyctl launch` (auto-detects Dockerfile)

---

### Option 6: VPS (Virtual Private Server)

Deploy to any VPS provider (DigitalOcean Droplet, Linode, Vultr, etc.).

#### Steps:

1. **Set Up VPS**
   - Create Ubuntu 22.04 server
   - SSH into server: `ssh root@your-server-ip`

2. **Install Dependencies**
   ```bash
   # Update system
   apt-get update && apt-get upgrade -y
   
   # Install Python and pip
   apt-get install -y python3 python3-pip python3-venv
   
   # Install FFmpeg
   apt-get install -y ffmpeg
   
   # Install Nginx (reverse proxy)
   apt-get install -y nginx
   ```

3. **Clone and Set Up App**
   ```bash
   # Clone your repository
   git clone https://github.com/yourusername/sonemote-project.git
   cd sonemote-project
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install gunicorn
   ```

4. **Set Up Gunicorn**
   Create `gunicorn_config.py`:
   ```python
   bind = "127.0.0.1:5000"
   workers = 4
   timeout = 120
   ```

5. **Create Systemd Service**
   Create `/etc/systemd/system/sonemote.service`:
   ```ini
   [Unit]
   Description=Sonemote Web App
   After=network.target

   [Service]
   User=www-data
   Group=www-data
   WorkingDirectory=/root/sonemote-project
   Environment="PATH=/root/sonemote-project/venv/bin"
   ExecStart=/root/sonemote-project/venv/bin/gunicorn -c gunicorn_config.py app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

6. **Start Service**
   ```bash
   systemctl daemon-reload
   systemctl start sonemote
   systemctl enable sonemote
   ```

7. **Configure Nginx**
   Create `/etc/nginx/sites-available/sonemote`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

8. **Enable Site**
   ```bash
   ln -s /etc/nginx/sites-available/sonemote /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

9. **Set Up SSL (Optional but Recommended)**
   ```bash
   apt-get install certbot python3-certbot-nginx
   certbot --nginx -d your-domain.com
   ```

---

## Important Configuration Changes for Production

### 1. Update app.py for Production

Modify the last lines of `app.py`:

```python
if __name__ == '__main__':
    init_model()
    
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(debug=False, host=host, port=port, threaded=True)
```

### 2. Use Gunicorn for Production

Install gunicorn:
```bash
pip install gunicorn
```

Update `requirements.txt`:
```
gunicorn>=21.2.0
```

Run with gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3. Set Environment Variables

- `PORT`: Port number (usually set by platform)
- `HOST`: Host address (usually `0.0.0.0`)
- `FFMPEG_LOCATION`: Path to FFmpeg executable (if not in PATH)

---

## Troubleshooting FFmpeg Issues

### Error: "ffprobe and ffmpeg not found"

#### Solution 1: Install FFmpeg on Windows (Detailed Steps)

**Important**: You need the **Windows executable build**, NOT the source code!

1. **Download FFmpeg Windows Executable**
   - Go to https://www.gyan.dev/ffmpeg/builds/ (recommended, easier to use)
   - Or go to https://ffmpeg.org/download.html → Windows builds → gyan.dev
   - Download **"ffmpeg-release-essentials.zip"** or **"ffmpeg-release-full.zip"**
   - **DO NOT** download "ffmpeg-release.zip" (that's the source code, not executable)
   - The file should be around 50-100 MB (not a few MB)

2. **Extract the ZIP File**
   - Right-click the downloaded ZIP file → "Extract All..."
   - Choose a location like `C:\ffmpeg` or `C:\Program Files\ffmpeg`
   - After extraction, you should see:
     ```
     C:\ffmpeg\
     ├── bin\
     │   ├── ffmpeg.exe      ← These are the executables you need
     │   ├── ffprobe.exe     ←
     │   └── ...
     ├── doc\
     ├── presets\
     └── ...
     ```
   - The `bin` folder contains the executable files

3. **Add FFmpeg to Windows PATH**
   
   **Method A: Using System Settings (Recommended)**
   - Press `Windows Key + X` and select **"System"**
   - Click **"Advanced system settings"** on the left sidebar
   - Click **"Environment Variables"** button at the bottom
   - Under **"System variables"** section, find and select **"Path"**
   - Click **"Edit..."** button
   - Click **"New"** button
   - Enter the path to the `bin` folder: `C:\ffmpeg\bin` (adjust if you used a different location)
   - Click **"OK"** on all dialogs to save
   - **Important**: Close and reopen your Command Prompt/PowerShell/IDE for changes to take effect

   **Method B: Using Command Prompt (Advanced)**
   ```cmd
   setx PATH "%PATH%;C:\ffmpeg\bin" /M
   ```
   Note: Requires Administrator privileges. Replace `C:\ffmpeg\bin` with your actual path.

4. **Verify Installation**
   - Open a **NEW** Command Prompt or PowerShell window (must be new to pick up PATH changes)
   - Type: `ffmpeg -version`
   - Type: `ffprobe -version`
   - Both commands should show version information
   - If you see "not recognized" error:
     - Verify you added the correct `bin` folder path (not the parent folder)
     - Make sure you restarted your terminal/IDE
     - Check that `ffmpeg.exe` and `ffprobe.exe` exist in that folder

5. **Alternative: Set FFMPEG_LOCATION Environment Variable**
   If you don't want to modify PATH, you can set a custom location:
   - Press `Windows Key + X` → **"System"** → **"Advanced system settings"** → **"Environment Variables"**
   - Under **"System variables"**, click **"New..."**
   - Variable name: `FFMPEG_LOCATION`
   - Variable value: `C:\ffmpeg\bin\ffmpeg.exe` (full path to ffmpeg.exe)
   - Click **"OK"** to save
   - Restart your application

#### Solution 2: Install FFmpeg on Mac/Linux

**Mac:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Linux (Fedora/CentOS):**
```bash
sudo yum install ffmpeg
```

#### Solution 3: Verify FFmpeg Installation

Test that FFmpeg is accessible:
```bash
# Windows (Command Prompt or PowerShell)
ffmpeg -version
ffprobe -version

# Mac/Linux
ffmpeg -version
ffprobe -version
```

Both commands should display version information. If not, check your PATH or environment variables.

#### Solution 4: For Cloud Platforms

- **Docker**: Use Dockerfile (includes FFmpeg installation) - see Option 5 above
- **Render/Railway**: Use buildpacks or Dockerfile that installs FFmpeg
- **Heroku**: Use `Aptfile` with `ffmpeg` entry
- **VPS**: Install via package manager (`apt-get install ffmpeg`)

#### Common Windows Issues:

**Issue**: "ffmpeg is not recognized as an internal or external command"
- **Fix**: You didn't add FFmpeg to PATH correctly, or you need to restart your terminal
- **Check**: Open File Explorer and verify `C:\ffmpeg\bin\ffmpeg.exe` exists

**Issue**: "The system cannot find the path specified"
- **Fix**: You added the wrong path. Make sure it's the `bin` folder, not the parent folder
- **Example**: Use `C:\ffmpeg\bin`, NOT `C:\ffmpeg`

**Issue**: Works in Command Prompt but not in IDE/Python
- **Fix**: Restart your IDE (VS Code, PyCharm, etc.) after adding to PATH
- **Alternative**: Set `FFMPEG_LOCATION` environment variable in your IDE settings

---

## Security Considerations

1. **Disable Debug Mode**: Set `debug=False` in production
2. **Use HTTPS**: Always use SSL/TLS certificates
3. **Set Secret Key**: Add Flask secret key:
   ```python
   app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production')
   ```
4. **Rate Limiting**: Consider adding rate limiting for API endpoints
5. **CORS**: Configure CORS properly (currently allows all origins)

---

## Testing Your Deployment

1. **Check Health Endpoint**: Create a simple health check:
   ```python
   @app.route('/health')
   def health():
       return jsonify({'status': 'ok'})
   ```

2. **Test Emotion Detection**: Use the web interface to test camera and emotion detection

3. **Test YouTube Downloads**: Try downloading a song to verify FFmpeg works

4. **Monitor Logs**: Check platform logs for errors

---

## Quick Reference: Platform Comparison

| Platform | Free Tier | FFmpeg Support | Difficulty | Best For |
|----------|-----------|----------------|------------|----------|
| Render | ✅ (with limits) | ⚠️ Needs config | Easy | Quick deployment |
| Railway | ✅ | ✅ Built-in | Easy | Modern apps |
| Heroku | ❌ (paid only) | ✅ Via buildpack | Medium | Established platform |
| DigitalOcean | ❌ | ⚠️ Needs config | Medium | Production apps |
| Docker | N/A | ✅ Via Dockerfile | Medium | Flexibility |
| VPS | ❌ | ✅ Full control | Hard | Full control |

---

## Next Steps

1. Choose a deployment platform
2. Follow the platform-specific steps above
3. Test your deployed application
4. Share your public URL with users!

For questions or issues, check the main README.md or WEBAPP_README.md files.

