# Vercel Deployment Instructions

This guide explains how to deploy the Vihangam Disaster Dashboard to Vercel without WebSockets.

## Prerequisites

1. A Vercel account
2. Git installed on your machine
3. The Vercel CLI (optional)

## Deployment Steps

### 1. Prepare Your Environment File

Create a `.env` file in the root directory with the following variables:

```
SECRET_KEY=your-secret-key
DEBUG=False
```

### 2. Push to GitHub

1. Create a GitHub repository
2. Push your code to the repository:
   ```
   git init
   git add .
   git commit -m "Initial commit for Vercel deployment"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### 3. Deploy to Vercel

#### Option 1: Using Vercel Dashboard

1. Log in to your Vercel account
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: Leave as default
   - Output Directory: Leave as default
5. Add environment variables from your `.env` file
6. Click "Deploy"

#### Option 2: Using Vercel CLI

1. Install Vercel CLI: `npm i -g vercel`
2. Log in to Vercel: `vercel login`
3. Deploy: `vercel`
4. Follow the prompts to configure your project

## Important Notes

1. This deployment is configured without WebSockets support
2. Static files are handled by the build_files.sh script
3. The optimized requirements-vercel.txt file is used to reduce deployment size
4. Database: Vercel deployments use SQLite by default, which is reset on each deployment. For production, consider using a persistent database service.

## Troubleshooting

If you encounter issues:

1. Check Vercel logs in the dashboard
2. Ensure all environment variables are set correctly
3. Verify the build_files.sh script has execute permissions
4. Make sure your Django settings are compatible with Vercel's serverless environment

## Limitations

1. No WebSocket support (real-time features disabled)
2. Limited file system access (uploads will not persist)
3. Function execution time limits (max 10-60 seconds depending on plan)