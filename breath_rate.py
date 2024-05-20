import cv2

class BreathDetector:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.prev_frame = None
        self.breath_count = 0

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def detect_breath(self, prev_frame, frame):
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray1 is None or gray2 is None:
            return 0

        corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)

        if corners is None:
            return 0

        corners_new, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, **self.lk_params)

        good_old = corners[status == 1]
        good_new = corners_new[status == 1]
        diff = good_new - good_old
        movement = cv2.norm(diff, cv2.NORM_L2)

        if movement > 25:
            return 1
        else:
            return 0

    def run_detection(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.prev_frame is not None:
                self.breath_count += self.detect_breath(self.prev_frame, frame)

            self.prev_frame = frame.copy()

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        print("Total number of breaths detected:", self.breath_count)

if __name__ == "__main__":
    video_source = 'test.mp4'
    detector = BreathDetector(video_source)
    detector.run_detection()
