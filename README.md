---
title: DeepFake Mirror
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
pinned: false
license: mit
short_description: Photo Exposure Check & Adversarial Protection Demo
---

# DeepFake Mirror | Photo Exposure Check

An experimental AI safety research tool that checks facial visibility, creates an experimental protected copy, and inspects what changed in the test face-recognition model.

## 🚀 Live Demo Workflow

1. **Upload** a face photo (JPG/PNG).
2. **Analyze** your visual exposure score with plain-English factor checks.
3. **Protect & Verify** — select protection strength ($\epsilon$) and inspect the measured embedding distance proof.
4. **Download** the protected PNG copy.

## 🔒 Session Privacy

- Photos are processed in-memory for the active session only.
- No database, no user accounts, no permanent storage.