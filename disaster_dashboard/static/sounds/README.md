# Audio Files Directory

## Required Files

### alert.mp3
This directory should contain an `alert.mp3` file for emergency alert notifications.

**Requirements:**
- Format: MP3
- Duration: 2-5 seconds recommended
- Volume: Medium intensity (the JavaScript sets volume to 0.7)
- Sound type: Alert tone, beep, or similar notification sound

**Usage:**
The file is used by the emergency alert system in the dashboard when critical alerts are triggered.

**Fallback:**
If the audio file is not available, the system will fall back to visual feedback (red flash animation).

**Example sources for alert sounds:**
- https://freesound.org/ (free sound effects)
- https://mixkit.co/free-sound-effects/ (royalty-free sounds)
- Record a simple beep or tone

## File Structure
```
sounds/
├── README.md (this file)
└── alert.mp3 (place your alert sound here)
```