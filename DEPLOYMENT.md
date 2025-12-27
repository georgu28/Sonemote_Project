# Sonemote Web App - Render Deployment Guide

This guide covers how to deploy the Sonemote web application to Render.

## Prerequisites

Before deploying, ensure you have:
- Trained model file (`emotion_model.h5`) in the project root
- Python 3.8+ installed locally
- All dependencies installed (`pip install -r requirements.txt`)
- A Render account (sign up at https://render.com)
- Git repository with your code pushed to GitHub/GitLab/Bitbucket

## Important Notes

- **Model File**: The `emotion_model.h5` file must be committed to your Git repository (not in `.gitignore`). The file is ~30MB, which is acceptable for GitHub (limit is 100MB).
- **Free Tier Limitations**: Render's free tier spins down after 15 minutes of inactivity. The first request after spin-down may take 30-60 seconds to respond.
- **Memory**: Render free tier includes 512MB RAM, which is sufficient for the emotion detection model.

## Deployment Steps

### 1. Prepare Your Repository

1. **Ensure Model File is Committed**
   ```bash
   # Check if emotion_model.h5 is in .gitignore
   cat .gitignore | grep emotion_model
   
   # If it's in .gitignore, remove it:
   # Edit .gitignore and remove or comment out: emotion_model.h5
   
   # Add and commit the model file
   git add emotion_model.h5
   git commit -m "Add trained emotion model for deployment"
   git push
   ```

2. **Verify Required Files Exist**
   - `app.py` - Main Flask application
   - `render.yaml` - Render configuration
   - `emotion_model.h5` - Trained emotion detection model
   - `requirements.txt` - Python dependencies
   - `templates/index.html` - Frontend template
   - `static/` - CSS and JavaScript files

### 2. Create Render Account

1. Go to https://render.com
2. Sign up for a free account (or use GitHub/GitLab to sign in)
3. Verify your email if required

### 3. Deploy via Render Dashboard

1. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab/Bitbucket account if not already connected
   - Select your Sonemote repository

2. **Configure Service**
   - **Name**: `sonemote-app` (or your choice)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (project root)
   - **Build Command**: Render will use `render.yaml` configuration
   - **Start Command**: Render will use `render.yaml` configuration

3. **Set Environment Variables**
   - Click "Advanced" → "Environment Variables"
   - Add if needed:
     - `FLASK_DEBUG`: `False` (for production)
     - `PORT`: Render sets this automatically (don't override)

4. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your app
   - Monitor the build logs for any errors
   - Your app will be available at: `https://sonemote-app.onrender.com` (or your chosen name)

### 4. Using render.yaml (Recommended)

The `render.yaml` file automates the deployment configuration:

```yaml
services:
  - type: web
    name: sonemote-app
    env: python
    plan: free
    buildCommand: |
      apt-get update
      apt-get install -y ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
      pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --log-level debug --access-logfile - --error-logfile -
```

This configuration:
- Installs FFmpeg for YouTube downloads
- Installs OpenCV system dependencies
- Uses Gunicorn with optimized settings for the emotion detection model
- Sets appropriate timeouts for model loading

## Configuration Details

### Gunicorn Settings

The start command uses these optimized settings:
- `--workers 1`: Single worker to reduce memory usage
- `--threads 4`: Multiple threads for concurrent requests
- `--timeout 120`: 2-minute timeout for model initialization
- `--log-level debug`: Detailed logging for troubleshooting

### System Dependencies

The build command installs:
- `ffmpeg`: Required for YouTube audio downloads
- `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1`: OpenCV dependencies

## Troubleshooting

### Issue: Model Not Found

**Symptoms:**
- App loads but emotion detection fails
- Logs show "Model not found" or "Creating untrained model structure"

**Solution:**
1. Verify `emotion_model.h5` is committed to Git:
   ```bash
   git ls-files | grep emotion_model.h5
   ```
2. Check file size (should be ~30MB)
3. Ensure file is in project root, not in a subdirectory
4. Redeploy on Render (trigger manual deploy from dashboard)

### Issue: Face Detection Not Working

**Symptoms:**
- Faces detected but emotions not predicted
- "Face cascade not initialized" errors

**Solution:**
1. Check Render build logs for OpenCV installation errors
2. Verify OpenCV system dependencies are installed (in `render.yaml`)
3. Verify `opencv-python-headless` is in `requirements.txt`
4. Check logs for cascade loading errors

### Issue: Worker Crashes or Timeouts

**Symptoms:**
- App crashes during startup
- "Worker timeout" errors
- Out of memory errors

**Solution:**
1. Model initialization happens at startup (may take 30-60 seconds)
2. Ensure `--timeout 120` in start command (already configured)
3. Check Render logs for specific error messages
4. Verify model file is not corrupted
5. Consider upgrading to paid plan for more resources

### Issue: Slow First Request

**Symptoms:**
- First request after spin-down takes 30-60 seconds

**Solution:**
- This is normal for Render free tier (cold start)
- Subsequent requests will be faster
- Consider upgrading to paid plan to avoid spin-downs
- Or use a service like UptimeRobot to ping your app every 5 minutes

### Issue: YouTube Downloads Not Working

**Symptoms:**
- YouTube download feature fails
- "FFmpeg not found" errors

**Solution:**
1. Verify FFmpeg is installed in build command (already in `render.yaml`)
2. Check build logs to confirm FFmpeg installation
3. Verify `yt-dlp` is in `requirements.txt`
4. Test FFmpeg availability in Render shell (if available)

## Environment Variables

Set these in Render dashboard under Environment:

- `FLASK_DEBUG`: Set to `False` for production
- `PORT`: Don't set this - Render sets it automatically

## Testing Your Deployment

1. **Check Health Endpoint**
   - Visit: `https://your-app.onrender.com/api/health`
   - Should return JSON with model and cascade status

2. **Test Emotion Detection**
   - Visit: `https://your-app.onrender.com`
   - Allow camera access
   - Test face detection and emotion prediction

3. **Monitor Logs**
   - Go to Render dashboard → Your Service → Logs
   - Check for errors during model initialization
   - Monitor for any runtime errors

## Updating Your Deployment

After making changes to your code:

1. **Commit and Push**
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```

2. **Render Auto-Deploys**
   - Render automatically deploys on push to the connected branch
   - Or manually trigger from dashboard: "Manual Deploy" → "Deploy latest commit"

3. **Monitor Deployment**
   - Watch build logs in Render dashboard
   - Verify deployment completes successfully

## Render Plan Comparison

| Feature | Free | Starter ($7/month) | Standard ($25/month) |
|---------|------|-------------------|---------------------|
| RAM | 512 MB | 512 MB | 2 GB |
| CPU | Shared | Shared | Dedicated |
| Spin-down | 15 min | Never | Never |
| Bandwidth | 100 GB | 100 GB | 1 TB |
| Auto-deploy | Yes | Yes | Yes |

**Recommendation**: Free tier is sufficient for testing. For production, consider Starter plan to avoid spin-downs.

## Security Considerations

1. **Disable Debug Mode**: Ensure `FLASK_DEBUG=False` in production
2. **HTTPS**: Render provides SSL certificates automatically
3. **CORS**: Currently allows all origins - configure for production if needed
4. **Rate Limiting**: Consider adding rate limiting for API endpoints
5. **Secret Keys**: Add Flask secret key via environment variable if using sessions

## Limitations

- **Free Tier Spin-down**: App spins down after 15 minutes of inactivity
- **Cold Starts**: First request after spin-down may take 30-60 seconds
- **Memory**: Free tier has 512MB RAM (sufficient but may be tight)
- **Build Time**: Free tier has longer build times

## Next Steps

1. Deploy to Render following the steps above
2. Test your deployed application
3. Monitor logs for any errors
4. Consider upgrading to paid plan for production use
5. Share your public URL with users!

For questions or issues, check the main README.md file or Render documentation.

