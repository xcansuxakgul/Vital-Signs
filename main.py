import threading
from temperature_detection import TemperatureDetection, VideoProcessor, ImageProcessor
from breath_rate import BreathDetector

class TemperatureThread(threading.Thread):
    def __init__(self, detection_obj):
        super().__init__()
        self.detection_obj = detection_obj
        
    def run(self):
        self.detection_obj.detect_fever()

class BreathThread(threading.Thread):
    def __init__(self, detection_obj):
        super().__init__()
        self.detection_obj = detection_obj
        
    def run(self):
        self.detection_obj.detect_breath()

# ornek kullanim
if __name__ == "__main__":
    detection_obj = TemperatureDetection(threshold=200, area_of_box=700, min_temp=102, font_scale_caution=1, font_scale_temp=0.7)

    fever_thread = TemperatureThread(detection_obj)
    breath_thread = BreathThread(detection_obj)


    fever_thread.start()
    breath_thread.start()

    fever_thread.join()
    breath_thread.join()

    print("All processing complete.")
