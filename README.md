# SONA AI

SONA AI is a machine learning project for detecting emotions from voice recordings. It uses audio features extracted with Librosa and trains a custom neural network to classify emotions such as euphoric, joyfully, sad, and surprised.

## Overview

This project is built to explore real-time audio classification using Python and deep learning. It includes raw audio data, preprocessing scripts, and trained model files. The goal is to build a system that can identify the emotional tone of a voice input.

## Features

- Processes raw audio files into features using Librosa
- Splits data into train, validation, and test sets
- Trains models including logistic regression and random forest
- Saves predictions and evaluation metrics
- Structure ready for real-time or batch emotion classification

## Project Structure
SONA AI/
├── data/
│   ├── raw/                  # Original audio and metadata
│   ├── processed/            # CSVs with processed features
│   ├── features/             # Numpy arrays for each emotion
│   └── models/               # Trained model files (.pkl)
├── src/
│   ├── config.py             # Project config and paths
│   ├── dataset.py            # Data loading and splits
│   ├── preprocess.py         # Audio cleaning and feature extraction
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation script
├── requirements.txt
└── README.md

## Getting Started

1. Install dependencies

```bash 
pip install -r requirements.txt
```
2.	Make sure your audio files are in data/raw/audio/
3.	Run preprocessing and feature extraction  
python src/preprocess.py

4.	Train the model
python src/train.py

5.	Evaluate results
python src/evaluate.py

Notes

Some model files and audio clips are already included. You can replace or expand the dataset by adding new .wav files in the same format.

Future Plans
	•	Add real-time microphone input for live emotion prediction
	•	Try other classifiers like CNN or RNN
	•	Add support for more emotion labels
