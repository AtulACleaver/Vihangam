# Heroku Deployment Notes

## ✅ Full Django + WebSocket Support

**Great News!** Heroku fully supports Django Channels and WebSocket functionality. Your real-time disaster detection features will work perfectly!

## Deployment Configuration

### Procfile
- Uses Daphne ASGI server for full WebSocket support
- Includes release phase for automatic migrations and static file collection

### Database
- Automatically configured for Heroku PostgreSQL
- Falls back to SQLite for local development
- Uses `dj-database-url` for seamless configuration

### Redis Channel Layer
- Configured to use Heroku Redis add-on for WebSocket channel layers
- Falls back to in-memory channel layer for development

## Required Environment Variables

Set these in your Heroku app settings:

```bash
SECRET_KEY=your-production-secret-key
DEBUG=False
DATABASE_URL=postgresql://... (automatically set by Heroku)
REDIS_URL=redis://... (set by Redis add-on)
```

## Add-ons Required

1. **Heroku Postgres** (for database)
2. **Heroku Redis** (for WebSocket channel layers)

## Static Files

Static files are handled by WhiteNoise middleware and served efficiently.
