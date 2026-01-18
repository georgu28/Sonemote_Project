# Sonemote Deployment Guide

## Project Overview

Flask web application with TensorFlow emotion detection model (~30MB). Detects emotions from webcam and plays matching music.

**Stack:** Flask, TensorFlow/Keras, OpenCV, Gunicorn  
**Memory:** ~400-600MB (model + dependencies)  
**Model:** `emotion_model.h5` (~30MB)

## Main Issues & Solutions

### 1. Worker Timeouts

**Problem:** Model initialization takes 30-60 seconds. Gunicorn timeout was too low.

**Solution:** Updated `render.yaml` with:
- `--timeout 180` (3 minutes)
- `--worker-timeout 180`
- `--preload` (load model once)

### 2. Memory Constraints

**Render Free Tier:** 512MB RAM (sufficient with optimizations)  
**Issue:** Spins down after 15 min inactivity (cold starts)

**Fix:** Single worker (`--workers 1`), preload model, monitor usage.

### 3. OpenCV Dependencies

**Required:** System libraries must be installed during build.

**Solution:** Already in `render.yaml` buildCommand:
```yaml
apt-get install -y ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

## Deployment: Render

**Configuration (`render.yaml`):**
- Timeout: 180s
- Workers: 1
- Threads: 4
- Preload: Enabled

**Before Deploying:**
1. Verify model is committed:
   ```bash
   git ls-files | grep emotion_model.h5
   ```

2. If missing:
   ```bash
   git add emotion_model.h5
   git commit -m "Add emotion model"
   git push
   ```

**Deployment:**
- Push to GitHub
- Render auto-deploys
- Monitor logs for "Model and face cascade loaded successfully!"

**Prevent Spin-Down (Optional):**
- Use UptimeRobot to ping `/api/health` every 5 minutes

## Verification

After deployment, check:

1. **Health Endpoint:**
   ```
   https://your-app.onrender.com/api/health
   ```
   Should show: `model_loaded: true`, `face_cascade_loaded: true`

2. **Face Detection:**
   - Open app in browser
   - Grant camera permission
   - Verify "Face Detected" appears

3. **Logs:**
   - "Model and face cascade loaded successfully!"
   - No "WORKER TIMEOUT" errors

## Troubleshooting

**Face detection not working:**
- Check `/api/health` endpoint
- Verify model initialized in logs
- Check browser console for API errors

**Worker timeouts:**
- Already fixed with 180s timeout
- If persists, check logs for initialization errors

**Memory issues:**
- Monitor Render dashboard (should stay under 512MB)
- Reduce `--threads` to 2 if needed

## Alternative: Render Starter ($7/month)

Same configuration but:
- No spin-down
- Better performance
- Recommended for production
