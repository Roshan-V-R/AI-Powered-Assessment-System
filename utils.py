from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

def evaluate_answer_with_feedback(expected: str, student_answer: str) -> Tuple[float, str]:
    """Evaluate answer using TF-IDF similarity and generate detailed feedback"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([expected, student_answer])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
    marks = similarity_score * 100
    
    feedback = "\nFeedback:\n"
    if similarity_score >= 0.8:
        feedback += "Excellent answer! Your response covers the key points well."
    elif similarity_score >= 0.6:
        feedback += "Good answer, but there's room for improvement. Consider including more details."
    elif similarity_score >= 0.4:
        feedback += "Your answer shows some understanding, but misses several important points."
    else:
        feedback += "Your answer needs improvement. Please review the topic and try again."
    
    model_keywords = set(expected.lower().split()) - set(['a', 'the', 'is', 'are', 'in', 'on', 'at', 'and', 'or'])
    student_keywords = set(student_answer.lower().split())
    missed_keywords = model_keywords - student_keywords
    
    if missed_keywords:
        feedback += "\n\nConsider including these key concepts in your answer:\n"
        feedback += ", ".join(missed_keywords)
    
    return marks, feedback