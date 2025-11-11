# 🚀 Deploy to Render.com - Simple Guide

**Time needed: 20 minutes**

---

## Step 1: Push Code to GitHub

### 1.0 Install Git (If Not Installed)

**Check if Git is installed:**

```powershell
git --version
```

**If you see an error "git is not recognized":**

1. Download Git: **https://git-scm.com/download/win**
2. Run the installer
3. Click "Next" on everything (use default settings)
4. **IMPORTANT:** Restart PowerShell after installation
5. Test again: `git --version`

✅ Should show: `git version 2.x.x`

### 1.1 Create GitHub Repository

1. Go to: **https://github.com/new**
2. Repository name: `iot-ml-monitoring`
3. Make it **PUBLIC** (required for free Render)
4. **DO NOT** check "Initialize with README"
5. Click **"Create repository"**

### 1.2 Push Your Code

Open PowerShell and run:

```powershell
cd "c:\Users\Admin.DESKTOP-QKNRRPJ\OneDrive - King Suad University\MLProject"

git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/iot-ml-monitoring.git
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your GitHub username!

**If it asks for password:** Use a Personal Access Token instead:
- Go to: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Check "repo" box
- Copy token and use it as password

✅ Code is now on GitHub!

---

## Step 2: Deploy Flask API

### 2.1 Sign Up for Render

1. Go to: **https://render.com/**
2. Click **"Get Started for Free"**
3. Sign up with **GitHub** (easiest)
4. Authorize Render to access your repositories

### 2.2 Create Web Service for Flask

1. Click **"New +"** → **"Web Service"**
2. Find and click **"Connect"** next to `iot-ml-monitoring`
3. Fill in these settings:

```
Name: iot-ml-api
Region: Oregon (US West) or closest to you
Branch: main
Root Directory: (leave EMPTY)
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn flask_api:app
Instance Type: Free
```

4. Click **"Create Web Service"**

⏳ **Wait 5-10 minutes** - watch the logs

### 2.3 Get Your API URL

Once deployed (shows "Live" ✅), copy your URL:

```
https://iot-ml-api-XXXX.onrender.com
```

**📝 SAVE THIS!** You need it for Next.js

### 2.4 Test Flask API

Open browser:

```
https://iot-ml-api-XXXX.onrender.com/health
```

Should show: `{"status":"healthy","model_loaded":true}`

✅ Flask API is live!

---

## Step 3: Deploy Next.js Dashboard

### 3.1 Create Web Service for Next.js

1. In Render, click **"New +"** → **"Web Service"**
2. Connect to **SAME repository**: `iot-ml-monitoring`
3. Fill in these settings:

```
Name: iot-dashboard
Region: Same as Flask API
Branch: main
Root Directory: nextjs-app
Runtime: Node
Build Command: npm install && npm run build
Start Command: npm start
Instance Type: Free
```

### 3.2 Add Environment Variables

**IMPORTANT!** Scroll down to "Environment Variables" section.

Click **"Add Environment Variable"** and add these ONE BY ONE:

```
Key: MYSQL_HOST
Value: YOUR_AIVEN_HOST (from Aiven dashboard)

Key: MYSQL_USER
Value: avnadmin

Key: MYSQL_PASSWORD
Value: YOUR_AIVEN_PASSWORD (from Aiven dashboard)

Key: MYSQL_DATABASE
Value: defaultdb

Key: ML_API_URL
Value: https://iot-ml-api-XXXX.onrender.com
```

**⚠️ IMPORTANT:**
- Use YOUR Flask API URL (from Step 2.3)
- NO `/health` at the end
- NO trailing slash

### 3.3 Deploy

1. Click **"Create Web Service"**
2. ⏳ Wait 5-10 minutes
3. Once "Live" ✅, copy your dashboard URL:

```
https://iot-dashboard-XXXX.onrender.com
```

---

## Step 4: Test Your Live Dashboard!

### 4.1 Open Dashboard

Go to: `https://iot-dashboard-XXXX.onrender.com`

**First load takes 30-60 seconds** (free tier wakes up from sleep)

You should see:
- ✅ Dashboard statistics
- ✅ Add Sensor Reading form
- ✅ Recent Predictions table

### 4.2 Add Test Sensor Reading

1. Fill in the form:
   - Equipment ID: `1`
   - Temperature: `75`
   - Vibration: `2.5`
   - Pressure: `100`
   - Humidity: `45`

2. Click **"Add & Predict"**

3. Wait 2-3 seconds...

4. ✅ You should see prediction appear!

---

## 🎉 SUCCESS!

Your system is now LIVE and accessible worldwide! 🌍

**Your URLs:**
- 🤖 Flask API: `https://iot-ml-api-XXXX.onrender.com`
- 🌐 Dashboard: `https://iot-dashboard-XXXX.onrender.com`
- 🗄️ Database: Aiven (already set up)

**Share your dashboard URL with anyone!**

---

## ⚠️ Important Notes

### Free Tier Behavior

**Services "sleep" after 15 minutes of inactivity**
- First request takes 30-60 seconds to wake up
- This is NORMAL for free tier
- Not broken, just waking up!

### Keep Services Awake (Optional)

Use **UptimeRobot** to ping your services:

1. Go to: https://uptimerobot.com/
2. Sign up (free)
3. Add monitor:
   - Type: HTTP(s)
   - URL: `https://iot-ml-api-XXXX.onrender.com/health`
   - Interval: Every 5 minutes
4. This keeps your API awake!

---

## 🔄 How to Update Your Live Site

Made code changes? Just push to GitHub:

```powershell
cd "c:\Users\Admin.DESKTOP-QKNRRPJ\OneDrive - King Suad University\MLProject"
git add .
git commit -m "Your update message"
git push
```

Render automatically redeploys! ✨ (takes 5 minutes)

---

## 🐛 Common Issues

### Problem: Dashboard shows "Network Error"

**Fix:** Check `ML_API_URL` in Render environment variables
- Should be: `https://iot-ml-api-XXXX.onrender.com`
- NO `/health` at end
- NO trailing `/`

### Problem: "Model not found" error

**Fix:** Make sure `rf_model.pkl` and `scaler.pkl` are in your GitHub repo
```powershell
git add rf_model.pkl scaler.pkl
git commit -m "Add model files"
git push
```

### Problem: Can't connect to database

**Fix:** 
1. Verify Aiven service is "Running"
2. Check environment variables in Render are EXACTLY matching
3. Make sure `mysql.js` has SSL enabled (port 27906)

### Problem: Build fails

**Fix:** Check logs in Render dashboard
- Common: Missing dependencies in `package.json` or `requirements.txt`
- Make sure files exist in correct directories

---

## 💰 Cost

Everything is **100% FREE:**
- ✅ Render Flask API: Free tier
- ✅ Render Next.js: Free tier  
- ✅ Aiven MySQL: Free Hobbyist plan

**Total: $0/month** forever! 🎉

---

## ✅ Deployment Checklist

- [ ] Aiven MySQL database created and tables added
- [ ] Code pushed to GitHub (public repository)
- [ ] Flask API deployed to Render
- [ ] Flask API `/health` endpoint working
- [ ] Next.js deployed to Render
- [ ] Environment variables added correctly
- [ ] Dashboard loads in browser
- [ ] Can add sensor reading and see prediction
- [ ] Data appears in Aiven database

---

**🎊 You're done! Your IoT ML Monitoring System is live!** 🚀
