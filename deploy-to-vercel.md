# 🚀 Deploy to Vercel - Vihangam Disaster Dashboard

## ✅ Configuration Complete!

Your Django project has been configured for Vercel deployment without WebSocket support. Here's what was updated:

- ✅ **vercel.json** - Configured for WSGI deployment
- ✅ **Django settings** - Disabled Django Channels
- ✅ **requirements.txt** - Removed WebSocket dependencies
- ✅ **build.sh** - Simplified for serverless deployment

## 🚀 Deployment Steps

### 1. Install Vercel CLI

Open **Command Prompt as Administrator** (not PowerShell) and run:

```cmd
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

This will open your browser to authenticate with Vercel.

### 3. Deploy Your Project

In your project directory, run:

```bash
vercel
```

Follow the prompts:
- **Set up and deploy?** → Yes
- **Which scope?** → Your personal account
- **Link to existing project?** → No
- **Project name** → `vihangam-disaster-dashboard` (or your preferred name)
- **In which directory is your code located?** → `./` (current directory)

### 4. Set Environment Variables

After deployment, set these environment variables in your Vercel dashboard:

```bash
vercel env add SECRET_KEY
# Enter your secret key when prompted

vercel env add DEBUG
# Enter: false
```

### 5. Redeploy with Environment Variables

```bash
vercel --prod
```

## 🎯 What Works on Vercel

✅ **Django Admin Panel**  
✅ **Static File Serving**  
✅ **Database Operations** (SQLite)  
✅ **All Non-WebSocket Features**  
✅ **Dashboard Views**  
✅ **Detection Features** (without real-time updates)  
✅ **Pathfinding Visualization**  

## ❌ What's Disabled

❌ **WebSocket Connections**  
❌ **Real-time Updates**  
❌ **Django Channels Features**  

## 🔧 Alternative for Real-time Features

If you need real-time updates later, consider:

1. **AJAX Polling** - Periodically fetch updates from the server
2. **Server-Sent Events** - One-way real-time updates
3. **Deploy to Heroku/Railway** - For full WebSocket support

## 📱 Expected URL

Your app will be available at: `https://your-project-name.vercel.app`

## 🆘 Troubleshooting

If deployment fails:

1. Check the build logs in Vercel dashboard
2. Ensure all files are committed to git
3. Verify `SECRET_KEY` environment variable is set
4. Check that static files are being collected properly

---

## 🎉 Ready to Deploy!

Your Django disaster dashboard is ready for Vercel deployment without WebSocket functionality. Run the commands above to get it live!