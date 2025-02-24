import os
import json
from datetime import datetime

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