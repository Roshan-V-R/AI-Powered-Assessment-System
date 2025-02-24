# Assessment System

This project is an automated assessment tool designed to generate academic questions, capture student responses via audio and video, evaluate answers using AI, and store the results for review. It leverages the Groq AI platform for question generation and evaluation, with a fallback to TF-IDF similarity scoring if needed. The system is interactive, allowing users to answer multiple questions and review their performance history.

## Features

- **Question Generation**: Creates easy academic questions on topics like science, history, and math using Groq AI.
- **Audio/Video Capture**: Uses a webcam to display questions and a microphone to record spoken answers, with real-time feedback on the screen.
- **Answer Evaluation**: Assesses responses with Groq AI for detailed feedback and scores, falling back to TF-IDF similarity if Groq fails.
- **Result Storage**: Saves each answer, score, feedback, and model answer in JSON format for later review.
- **Multi-Question Support**: Allows users to specify how many questions they want to answer in a single session.
- **History Review**: Provides an option to view all previous answers for a given student.

## Prerequisites

To run this project, you'll need:
- **Python 3.x**: Ensure Python is installed (download from python.org if needed).
- **Git**: For cloning the repository (download from git-scm.com).
- **Groq API Key**: Sign up at groq.com to get an API key for AI functionality.
- **Webcam and Microphone**: Hardware required for capturing answers.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/assessment_system.git
   cd assessment_system
