from groq_assessment import GroqAssessment
from answer_recorder import AnswerRecorder
from audio_video import capture_video_and_audio

def main():
    print("\n=== Assessment Session Started ===")

    api_key = "Enter your api key"
    groq_assessment = GroqAssessment(api_key)
    recorder = AnswerRecorder()

    student_name = input("\nPlease enter your name: ")

    while True:
        try:
            num_quest = int(input("\nNo: of Questions ="))
            if num_quest >= 1:
                break
            else:
                print("Please enter a number of 1 or greater")
        except ValueError:
            print("Please enter a valid number")

    for i in range(num_quest):
        print(f"\n=== Question {i + 1} of {num_quest} ===")        


        question, model_answer = groq_assessment.generate_question()
        print(f"\nQuestion: {question}")

        print("\nStarting webcam...")
        student_answer = capture_video_and_audio(question)
        
        if student_answer:
            print("\nYour Answer: ", student_answer)
            print("\nEvaluating your answer...")

            marks, feedback = groq_assessment.evaluate_answer(model_answer, student_answer)
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
        
        else:
            print("\nNo answer was recorded.")

        if i < num_quest - 1:
            print("\n" + "-" * 50)    
   

    show_history = input("\nWould you like to see your previous answers? (y/n): ")
    if show_history.lower() == 'y':
        history = recorder.get_student_history(student_name)
        print("\n=== Previous Answers ===")
        for entry in history:
            print(f"\nDate: {entry['timestamp']}")
            print(f"Question: {entry['question']}")
            print(f"Marks: {entry['marks']:.2f}/100")    
        
    print("\n=== Assessment Session Ended ===")

if __name__ == "__main__":
    main()