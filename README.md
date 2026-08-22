# 🚦 SafeJunction AI

AI-Powered Traffic Junction Safety & Risk Monitoring System.

## 🌐 Live Demo

https://safejunction-ai-lwih8qkjhgm73gy2yghq2w.streamlit.app/

## 📌 Problem

Busy traffic junctions can become dangerous because of congestion, high pedestrian activity, heavy vehicles, and pedestrian-vehicle conflicts.

## 💡 Solution

SafeJunction AI analyzes traffic videos using computer vision to identify vehicles and pedestrians, track their movement, estimate traffic density, detect potential pedestrian-vehicle conflicts, and highlight risk hotspots.

## 🤖 Key Features

- 🚗 Vehicle detection and counting
- 🚶 Pedestrian detection and counting
- 🎯 Object tracking with tracking IDs
- 🚦 Traffic density analysis
- ⚠️ Pedestrian-vehicle interaction detection
- 🛡️ AI-based safety score
- 🗺️ Risk hotspot detection
- 🚨 Safety alerts
- 🎥 AI-processed tracking video
- 🌐 Public web deployment

## 🛠️ Technology Stack

- Python
- Streamlit
- YOLO
- OpenCV
- Ultralytics
- FFmpeg
- Python-LAP tracking dependency

## ▶️ Run Locally

```bash
git clone https://github.com/RishavPoray/SafeJunction-AI.git
cd SafeJunction-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py