#!/bin/bash

# Exit on error
set -e

# Install dependencies (using optimized requirements for Vercel)
echo "Installing dependencies..."
pip install -r requirements-vercel.txt

# Collect static files
echo "Collecting static files..."
cd disaster_dashboard
python manage.py collectstatic --noinput --clear

# Make directory for Vercel static files
echo "Copying static files for Vercel..."
mkdir -p ../staticfiles
if [ -d "staticfiles" ] && [ "$(ls -A staticfiles)" ]; then
    cp -r staticfiles/* ../staticfiles/
else
    echo "Warning: staticfiles directory is empty or doesn't exist"
fi

echo "Build completed successfully!"