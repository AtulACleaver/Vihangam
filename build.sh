#!/bin/bash

echo "Starting Vercel build process..."

# Change to the disaster_dashboard directory
cd disaster_dashboard

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Vercel build completed successfully!"
