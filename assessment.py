import cv2
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import threading
import queue
import json
from datetime import datetime
import groq
import os
from typing import Tuple
import joblib

class GroqAssessment:
    def __init__(self, api_key: str):
        """Initialize Groq client with API key"""
        self.client = groq.Client(api_key=api_key)
        
    def generate_question(self, topics=["science", "history", "math"]) -> Tuple[str, str]:
        """Generate a question and model answer using Groq"""
        topic = random.choice(topics)
        prompt = f"""Generate an easy academic question about {topic} and its detailed answer.
        Format the response as a JSON object with 'question' and 'answer' keys."""
        
        try:
            completion = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an educational assessment expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            response = json.loads(completion.choices[0].message.content)
            return response["question"], response["answer"]
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            # Fallback to a default question if Groq fails
            return ("What is the scientific method?", 
                   "The scientific method is a systematic approach to investigation consisting of observation, hypothesis, experimentation, analysis, and conclusion.")

    def evaluate_answer(self, model_answer: str, student_answer: str) -> Tuple[float, str]:  
        """Evaluate student answer using Groq"""
        prompt = f"""Evaluate this student answer and provide detailed feedback.
        
        Question's model answer: {model_answer}
        Student's answer: {student_answer}
        
        Provide evaluation as JSON with:
        - score (0-100)
        - detailed_feedback
        - missed_concepts
        """
        
        try:
            completion = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert educational assessor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            evaluation = json.loads(completion.choices[0].message.content)
            
            feedback = f"\nFeedback:\n{evaluation['detailed_feedback']}"
            if evaluation['missed_concepts']:
                feedback += "\n\nKey concepts to include:\n"
                feedback += ", ".join(evaluation['missed_concepts'])
                
            return float(evaluation['score']), feedback
            
        except Exception as e:
            print(f"Error in Groq evaluation: {str(e)}")
            # Fallback to TF-IDF similarity if Groq fails
            return evaluate_answer_with_feedback(model_answer, student_answer)

def evaluate_answer_with_feedback(expected: str, student_answer: str) -> Tuple[float, str]:
    """Evaluate answer using TF-IDF similarity and generate detailed feedback"""
    # Calculate similarity score
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([expected, student_answer])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
    marks = similarity_score * 100
    
    # Generate feedback
    feedback = "\nFeedback:\n"
    if similarity_score >= 0.8:
        feedback += "Excellent answer! Your response covers the key points well."
    elif similarity_score >= 0.6:
        feedback += "Good answer, but there's room for improvement. Consider including more details."
    elif similarity_score >= 0.4:
        feedback += "Your answer shows some understanding, but misses several important points."
    else:
        feedback += "Your answer needs improvement. Please review the topic and try again."
    
    # Add missed keywords analysis
    model_keywords = set(expected.lower().split()) - set(['a', 'the', 'is', 'are', 'in', 'on', 'at', 'and', 'or'])
    student_keywords = set(student_answer.lower().split())
    missed_keywords = model_keywords - student_keywords
    
    if missed_keywords:
        feedback += "\n\nConsider including these key concepts in your answer:\n"
        feedback += ", ".join(missed_keywords)
    
    return marks, feedback

class AnswerRecorder:
    def __init__(self, save_dir="student_answers"):
        """Initialize the answer recorder with a directory to save answers"""
        self.save_dir = save_dir
        self.ensure_save_directory()
        
    def ensure_save_directory(self):
        """Create the save directory if it doesn't exist"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def save_answer(self, student_name, question, student_answer, marks, feedback, model_answer):
        """Save a student's answer and assessment details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_name}_{timestamp}.json"
        
        answer_data = {
            "timestamp": timestamp,
            "student_name": student_name,
            "question": question,
            "student_answer": student_answer,
            "marks": marks,
            "feedback": feedback,
            "model_answer": model_answer
        }
        
        file_path = os.path.join(self.save_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(answer_data, f, indent=4)
            
        return file_path
        
    def get_student_history(self, student_name):
        """Retrieve all answers for a specific student"""
        student_answers = []
        
        for filename in os.listdir(self.save_dir):
            if filename.startswith(student_name) and filename.endswith('.json'):
                file_path = os.path.join(self.save_dir, filename)
                with open(file_path, 'r') as f:
                    answer_data = json.load(f)
                    student_answers.append(answer_data)
                    
        return sorted(student_answers, key=lambda x: x['timestamp'], reverse=True)

class AudioCaptureThread(threading.Thread):
    def __init__(self, audio_queue):
        threading.Thread.__init__(self)
        self.audio_queue = audio_queue
        self.running = True
        self.recognizer = sr.Recognizer()

    def run(self):
        with sr.Microphone() as source:
            print("\nAudio capture started. Speak your answer clearly...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=60)
                    try:
                        text = self.recognizer.recognize_google(audio)
                        self.audio_queue.put(text)
                        self.running = False  # Stop after getting one complete answer
                    except sr.UnknownValueError:
                        pass  # Continue listening if speech wasn't recognized
                except sr.WaitTimeoutError:
                    pass  # Continue listening if timeout occurs
                
    def stop(self):
        self.running = False

def capture_video_and_audio(question):
    """Integrated video display and audio capture"""
    cap = cv2.VideoCapture(0)
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        return None
    
    height, width = frame.shape[:2]
    
    # Initialize audio capture in separate thread
    audio_queue = queue.Queue()
    audio_thread = AudioCaptureThread(audio_queue)
    audio_thread.start()
    
    # Initialize answer state
    recording = True
    answer_text = ""
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Display question
        words = question.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 50:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y_pos = 30
        for line in lines:
            cv2.putText(frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
        
        # Check for completed answer in queue
        try:
            answer_text = audio_queue.get_nowait()
            recording = False
        except queue.Empty:
            pass
        
        # Display status and instructions
        if recording:
            cv2.putText(frame, "Recording... Speak your answer clearly", 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to finish answering", 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Answer recorded! Press 'q' to continue", 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        cv2.imshow('Assessment Session', frame)
        
        # Check for 'q' key or if answer is complete
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
   
    audio_thread.stop()
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    
    return answer_text


def save_assessment_model(groq_assessment, answer_recorder, filename="assessment_model.joblib"):
    """Save the assessment model components"""
    model = {
        "groq_assessment": groq_assessment,
        "answer_recorder": answer_recorder
    }
    joblib.dump(model, filename)
    print(f"Model saved as: {filename}")




def main():
    """Main function to run the automated assessment system"""
    print("\n=== Assessment Session Started ===")
    
    # Initialize Groq assessment with direct API key
    api_key = "Enter your API key"  
    groq_assessment = GroqAssessment(api_key)
    recorder = AnswerRecorder()
    
    # Get student name
    student_name = input("\nPlease enter your name: ")
    
    # Generate question using Groq
    question, model_answer = groq_assessment.generate_question()
    print(f"\nQuestion: {question}")
    
    print("\nStarting webcam...")
    student_answer = capture_video_and_audio(question)
    
    if student_answer:
        print("\nYour answer:", student_answer)
        print("\nEvaluating your answer...")
        
        # Evaluate using Groq
        marks, feedback = groq_assessment.evaluate_answer(model_answer, student_answer)
        
        # Save the answer and assessment
        saved_file = recorder.save_answer(
            student_name=student_name,
            question=question,
            student_answer=student_answer,
            marks=marks,
            feedback=feedback,
            model_answer=model_answer
        )
        
        print("\n=== Evaluation Results ===")
        print(f"\nMarks: {marks:.2f}/100")
        print(feedback)
        print("\nModel Answer:")
        print(model_answer)
        print(f"\nYour answer has been saved to: {saved_file}")
        
        # Show student history
        show_history = input("\nWould you like to see your previous answers? (y/n): ")
        if show_history.lower() == 'y':
            history = recorder.get_student_history(student_name)
            print("\n=== Previous Answers ===")
            for entry in history:
                print(f"\nDate: {entry['timestamp']}")
                print(f"Question: {entry['question']}")
                print(f"Marks: {entry['marks']:.2f}/100")
    else:
        print("\nNo answer was recorded.")
        
    print("\n=== Assessment Session Ended ===")
    #Saving the model
    save_assessment_model(groq_assessment, recorder)

if __name__ == "__main__":
    main()