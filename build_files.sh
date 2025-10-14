#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Collect static files
cd disaster_dashboard
python manage.py collectstatic --noinput --clear

# Make directory for Vercel static files
mkdir -p ../staticfiles_build
cp -r staticfiles/* ../staticfiles_build/