import cv2
import speech_recognition as sr
import threading
import queue

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
                        self.running = False
                    except sr.UnknownValueError:
                        pass
                except sr.WaitTimeoutError:
                    pass
                
    def stop(self):
        self.running = False

def capture_video_and_audio(question):
    """Integrated video display and audio capture"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        return None
    
    height, width = frame.shape[:2]
    audio_queue = queue.Queue()
    audio_thread = AudioCaptureThread(audio_queue)
    audio_thread.start()
    
    recording = True
    answer_text = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
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
        
        try:
            answer_text = audio_queue.get_nowait()
            recording = False
        except queue.Empty:
            pass
        
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    audio_thread.stop()
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    return answer_text