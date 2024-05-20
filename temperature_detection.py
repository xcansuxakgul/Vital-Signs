import os
import numpy as np
import cv2

class TemperatureDetection:
    def __init__(self, threshold=200, area_of_box=700, min_temp=102, font_scale_caution=1, font_scale_temp=0.7):
        self.threshold = threshold
        self.area_of_box = area_of_box
        self.min_temp = min_temp
        self.font_scale_caution = font_scale_caution
        self.font_scale_temp = font_scale_temp

    def convert_fahrenheit_to_celsius(self, temperature_fahrenheit):
        temperature_celsius = (temperature_fahrenheit - 32) * 5 / 9
        return round(temperature_celsius, 1)

    def convert_to_temperature(self, pixel_avg):
        """
        Converts pixel value (mean) to temperature (Fahrenheit) depending upon the camera hardware.
        """
        return pixel_avg / 2.25

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

        _, binary_thresh = cv2.threshold(heatmap_gray, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
        image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

        contours, _ = cv2.findContours(image_opening, 1, 2)

        image_with_rectangles = np.copy(heatmap)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if (w) * (h) < self.area_of_box:
                continue

            mask = np.zeros_like(heatmap_gray)
            cv2.drawContours(mask, contour, -1, 255, -1)

            mean = self.convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])
            temperature = round(mean, 2)
            color = (0, 255, 0) if temperature < self.min_temp else (255, 255, 127)

            if temperature >= self.min_temp:
                cv2.putText(image_with_rectangles, "High temperature detected !!!", (35, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_caution, color, 2, cv2.LINE_AA)

            image_with_rectangles = cv2.rectangle(
                image_with_rectangles, (x, y), (x+w, y+h), color, 2)

            cv2.putText(image_with_rectangles, "{} C°".format(self.convert_fahrenheit_to_celsius(temperature)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_temp, color, 2, cv2.LINE_AA)

        return image_with_rectangles

class Processor:
    def __init__(self, input_path, output_path, detection_obj, resize_factor=None):
        self.input_path = input_path
        self.output_path = output_path
        self.detection_obj = detection_obj
        self.resize_factor = resize_factor

    def process(self):
        raise NotImplementedError("process method must be implemented in subclasses")
    
class VideoProcessor(Processor):
    def __init__(self, input_path, output_path, detection_obj):
        super().__init__(input_path, output_path, detection_obj)

    def process(self):
        video = cv2.VideoCapture(self.input_path)
        video_frames = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            processed_frame = self.detection_obj.process_frame(frame)
            video_frames.append(processed_frame)

            cv2.imshow('frame', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        if video_frames:
            height, width, _ = video_frames[0].shape
            size = (width, height)
            out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'MJPG'), 100, size)

            for frame in video_frames:
                out.write(frame)
            out.release()


class ImageProcessor(Processor):
    def __init__(self, input_image_path, output_image_path, detection_obj, resize_factor=0.5):
        super().__init__(input_image_path, output_image_path, detection_obj, resize_factor)

    def process(self):
        img = cv2.imread(self.input_path)

        if img is None:
            raise ValueError("Image not found at the specified path: {}".format(self.input_path))

        processed_img = self.detection_obj.process_frame(img)
        
        if self.resize_factor:
            height, width, _ = processed_img.shape
            dim = (int(width * self.resize_factor), int(height * self.resize_factor))
            processed_img = cv2.resize(processed_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(self.output_path, processed_img)
        cv2.imshow('output', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ornek kullanim
if __name__ == "__main__":
    detection_obj = TemperatureDetection(threshold=200, area_of_box=700, min_temp=102, font_scale_caution=1, font_scale_temp=0.7)

    video_processor = VideoProcessor('sample_videos/sample_video.mp4', 'outputs/output_video.avi', detection_obj)
    video_processor.process()

    image_processor = ImageProcessor('sample_images/input_image.jpg', 'outputs/output_image.jpg', detection_obj, resize_factor=0.5)
    image_processor.process()