# Sonemote Web App - Vercel Deployment Guide

This guide covers how to deploy the Sonemote web application to Vercel using the Hobby plan.

## Prerequisites

Before deploying, ensure you have:
- Trained model file (`emotion_model.h5`) in the project root
- Python 3.11+ installed locally
- All dependencies installed (`pip install -r requirements.txt`)
- A Vercel account (sign up at https://vercel.com)
- Git repository with your code pushed to GitHub/GitLab/Bitbucket

## Important Notes

- **Model File**: The `emotion_model.h5` file must be committed to your Git repository (not in `.gitignore`). The file is ~30MB, which is acceptable for GitHub (limit is 100MB).
- **Memory Limits**: Vercel Hobby plan includes 1GB memory per function. The emotion detection model requires significant memory, so the configuration uses 3008MB (Pro plan feature). For Hobby plan, you may need to optimize or upgrade.
- **Function Timeout**: Vercel Hobby plan has a 10-second timeout for serverless functions. The Pro plan allows up to 60 seconds, which is recommended for this app.
- **FFmpeg**: YouTube download feature requires FFmpeg, which is not available in Vercel's serverless environment. This feature will not work on Vercel.

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
   - `api/index.py` - Vercel serverless function wrapper
   - `vercel.json` - Vercel configuration
   - `emotion_model.h5` - Trained emotion detection model
   - `requirements.txt` - Python dependencies
   - `templates/index.html` - Frontend template
   - `static/` - CSS and JavaScript files

### 2. Install Vercel CLI (Optional)

You can deploy via the Vercel dashboard or using the CLI:

```bash
npm install -g vercel
```

### 3. Deploy via Vercel Dashboard

1. **Go to Vercel Dashboard**
   - Visit https://vercel.com/dashboard
   - Click "Add New..." → "Project"

2. **Import Your Repository**
   - Connect your GitHub/GitLab/Bitbucket account
   - Select your Sonemote repository
   - Click "Import"

3. **Configure Project**
   - **Framework Preset**: Other (or leave as auto-detected)
   - **Root Directory**: `./` (project root)
   - **Build Command**: Leave empty (Vercel auto-detects Python)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`

4. **Set Environment Variables**
   - Click "Environment Variables"
   - Add if needed:
     - `PYTHON_VERSION`: `3.11` (optional, Vercel auto-detects)
     - `FLASK_DEBUG`: `False` (for production)

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Your app will be available at: `https://your-project-name.vercel.app`

### 4. Deploy via Vercel CLI

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy**
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Link to existing project or create new
   - Confirm settings
   - Deploy

3. **Production Deployment**
   ```bash
   vercel --prod
   ```

## Configuration Files

### vercel.json

The `vercel.json` file configures Vercel deployment:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/index.py"
    },
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.11"
  },
  "functions": {
    "api/index.py": {
      "maxDuration": 60,
      "memory": 3008
    }
  }
}
```

**Note**: The `maxDuration: 60` and `memory: 3008` require Vercel Pro plan. For Hobby plan, these will be limited to 10 seconds and 1024MB respectively.

### api/index.py

This file wraps the Flask app for Vercel's serverless environment:

```python
import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory so model file can be found
os.chdir(parent_dir)

# Import the Flask app
from app import app

# Export app for Vercel
__all__ = ['app']
```

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
4. Redeploy on Vercel

### Issue: Function Timeout

**Symptoms:**
- Requests timeout after 10 seconds
- Model initialization takes too long

**Solution:**
1. Model initialization happens on first request (lazy loading)
2. First request may timeout - subsequent requests should be faster
3. Consider upgrading to Vercel Pro plan for 60-second timeout
4. Optimize model loading (already done in code)

### Issue: Memory Limit Exceeded

**Symptoms:**
- Function crashes with memory errors
- "Out of memory" errors in logs

**Solution:**
1. Vercel Hobby plan has 1GB memory limit per function
2. The model requires significant memory
3. Upgrade to Vercel Pro plan for 3GB memory
4. Or optimize the model (quantization, smaller architecture)

### Issue: Face Detection Not Working

**Symptoms:**
- Faces detected but emotions not predicted
- "Face cascade not initialized" errors

**Solution:**
1. OpenCV dependencies are included in requirements.txt
2. Check Vercel build logs for OpenCV installation errors
3. Verify `opencv-python-headless` is in requirements.txt
4. Model initialization may fail silently - check logs

### Issue: YouTube Downloads Not Working

**Symptoms:**
- YouTube download feature fails
- "FFmpeg not found" errors

**Solution:**
- FFmpeg is not available in Vercel's serverless environment
- This feature will not work on Vercel
- Consider using a different deployment platform (Render, Railway) if YouTube downloads are required

## Vercel Plan Comparison

| Feature | Hobby (Free) | Pro ($20/month) |
|---------|--------------|-----------------|
| Function Timeout | 10 seconds | 60 seconds |
| Memory per Function | 1024 MB | 3008 MB |
| Bandwidth | 100 GB | 1 TB |
| Serverless Functions | Unlimited | Unlimited |
| Team Collaboration | Limited | Full |

**Recommendation**: For production use with emotion detection, Vercel Pro plan is recommended due to memory and timeout requirements.

## Environment Variables

Set these in Vercel dashboard under Project Settings → Environment Variables:

- `FLASK_DEBUG`: Set to `False` for production
- `PYTHON_VERSION`: `3.11` (optional, auto-detected)

## Testing Your Deployment

1. **Check Health Endpoint**
   - Visit: `https://your-app.vercel.app/api/health`
   - Should return JSON with model and cascade status

2. **Test Emotion Detection**
   - Visit: `https://your-app.vercel.app`
   - Allow camera access
   - Test face detection and emotion prediction

3. **Monitor Logs**
   - Go to Vercel dashboard → Your Project → Functions
   - Click on function invocations to see logs
   - Check for errors during model initialization

## Updating Your Deployment

After making changes to your code:

1. **Commit and Push**
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```

2. **Vercel Auto-Deploys**
   - Vercel automatically deploys on push to main/master branch
   - Or manually trigger from dashboard

3. **Preview Deployments**
   - Pull requests automatically get preview deployments
   - Test before merging to production

## Security Considerations

1. **Disable Debug Mode**: Ensure `FLASK_DEBUG=False` in production
2. **HTTPS**: Vercel provides SSL certificates automatically
3. **CORS**: Currently allows all origins - configure for production if needed
4. **Rate Limiting**: Consider adding rate limiting for API endpoints
5. **Secret Keys**: Add Flask secret key via environment variable if using sessions

## Limitations

- **YouTube Downloads**: Not supported (FFmpeg unavailable)
- **File Uploads**: Limited by Vercel's serverless function constraints
- **Cold Starts**: First request after inactivity may be slow (model loading)
- **Memory**: Hobby plan may be insufficient for model loading

## Next Steps

1. Deploy to Vercel following the steps above
2. Test your deployed application
3. Monitor function logs for errors
4. Consider upgrading to Pro plan for production use
5. Share your public URL with users!

For questions or issues, check the main README.md file or Vercel documentation.
