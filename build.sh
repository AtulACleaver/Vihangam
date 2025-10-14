#!/bin/bash

# Change to the disaster_dashboard directory
cd disaster_dashboard

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Create database tables (for SQLite)
python manage.py migrate --noinput

echo "Build completed successfully"