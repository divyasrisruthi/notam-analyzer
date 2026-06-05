# NOTAM Waypoint Analyzer

A web-based NOTAM (Notice to Airmen) analyzer that processes route segments, resolves waypoints, and visualizes them on an interactive map.

---

## 🚀 Overview

This application allows users to input NOTAM text and extract structured route information, including:

- Waypoints
- Coordinates
- Distance/direction references (e.g., "50NM WEST OF XYZ")
- Airway segments

The system resolves these using a CSV-based waypoint dataset and displays results on a map interface.

---

## ✅ Features

- ✅ NOTAM text parsing and segment extraction
- ✅ Waypoint resolution using CSV dataset
- ✅ KD-tree based nearest waypoint lookup
- ✅ Distance/direction handling (dist_dir logic)
- ✅ Interactive map visualization (Leaflet)
- ✅ Supports route segments, coordinates, and mixed formats
- ✅ Safe handling of malformed or partial data
- ✅ Automatic update via CSV file changes

---

## 🏗️ Project Structure

backend/       → Flask API (core logic)
frontend/      → UI (HTML, CSS, JS)
data/          → Waypoint dataset (wavepoints.csv)
requirements.txt → Python dependencies

---

## ⚙️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript (Leaflet.js)
- **Data**: CSV-based waypoint database
- **Processing**: KD-tree spatial indexing (SciPy, NumPy)

---

## 📦 Deployment

This application is designed to be deployed as a **Flask web service**.

### Hosting options:
- ✅ Render (recommended for auto-deploy via GitHub)
- ✅ PythonAnywhere (manual deployment option)

---

## 🔄 Updating Waypoint Data

The waypoint dataset is stored in:
data/wavepoints.csv

### To update:

1. Open `wavepoints.csv`
2. Replace or edit contents
3. Commit changes (if using Git)

Rendering platform behavior:
- ✅ Render → auto redeploy + reload dataset

The application is designed to:
- ✅ Handle minor formatting issues
- ✅ Skip invalid rows
- ✅ Retain previous dataset if new file is invalid

---

## ⚠️ Notes

- Only the waypoint CSV file should be modified during regular updates.
- Backend and frontend logic should not be edited unless making code changes.
- Free hosting tiers (e.g., Render) may introduce cold-start delays (~30–60 seconds after inactivity).

---

## 🔐 Environment Configuration

Environment variables are defined in:
.env.example

This file acts as a **template** and does not contain any sensitive information.

---

## 👥 Usage

1. Open the deployed web app  
2. Paste NOTAM text  
3. Submit analysis  
4. View extracted segments and map visualization  

---

## ✅ Status

✔ Functional NOTAM parsing  
✔ Waypoint resolution  
✔ Deployment-ready  
✔ Dataset update support  
