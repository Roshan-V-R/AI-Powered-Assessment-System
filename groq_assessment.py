import groq
import json
from typing import Tuple
from utils import evaluate_answer_with_feedback
import random

class GroqAssessment:
    def __init__(self,api_key: str):
        self.client = groq.Client(api_key = "Your_api_key")

    def generate_question(self, topics = ["science", "history", "general knowledge"]) -> Tuple[str, str]:
        topic = random.choice(topics)
        prompt = f"""Generate an easy academic question about {topic} and its detailed answer.
        Format the response as a JSON object with 'question' and 'answer' keys."""

        try:
            completion = self.client.chat.completions.create(
                model = "mixtral-8x7b-32768",
                messages = [{"role": "system", "content": "You are an educational assessment expert."},
                            {"role": "user", "content": prompt}],
                temperature = 0.7
                )
            response = json.load(completion.choices[0].message.content)
            return response ["question", response["answer"]]
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            return ("What is the scientific method?", 
                   "The scientific method is a systematic approach to investigation consisting of observation, hypothesis, experimentation, analysis, and conclusion.")
        
    def evaluate_answer(self, model_answer: str, student_answer: str) -> Tuple[float,str]:
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
                model = "mixtral-8x7b-32768",
                messages = [{"role": "system", "content": "You are an expert educational assessor."},
                    {"role": "user", "content": prompt}
                ], temperature = 0.3
            )
            evaluation = json.loads(completion.choices[0].message.content)
            feedback = f"\nFeedback:\n{evaluation['detailed_feedback']}"
            if evaluation['missed_concepts']:
                feedback += "\n\nKey concepts to include:\n"
                feedback += ", ".join(evaluation['missed_concepts'])
            return float(evaluation['score']), feedback
        except Exception as e:
            print(f"Error in Groq evaluation: {str(e)}")
            return evaluate_answer_with_feedback(model_answer,student_answer)    