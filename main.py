from ultralytics import YOLO
import cv2 as cv

from sort.sort import *
from utils import *

# load models
model = YOLO('yolov8n.pt')
plate_detector_model = YOLO('./models/license_plate_detector.pt')

mot_tracker = Sort()

# load video
cap = cv.VideoCapture('./demo_1.mp4')

vehicle_ids = [2, 3, 5, 7]
results = {}

# read frames
frame_nmb = 0
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        results[frame_nmb] = {}
        # detect vehicles
        vehicles = model(frame)[0]
        vehicles_ = []
        for vehicle in vehicles.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = vehicle
            if int(class_id) in vehicle_ids:
                vehicles_.append([x1, y1, x2, y2, score])

        # track vehicles
        tracking_ids = mot_tracker.update(np.asarray(vehicles_))

        # detect plates
        plates = plate_detector_model(frame)[0]
        print(frame_nmb)

        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            # map plate -> car
            x1car, y1car, x2car, y2car, car_id = map_car(plate, tracking_ids)

            if car_id != -1:
                # crop plate
                cropped_plate = frame[int(y1): int(y2), int(x1): int(x2), :]

                # process plate
                cropped_plate_gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
                _, plate_thresholded = cv.threshold(cropped_plate_gray, 64, 255, cv.THRESH_BINARY_INV)

                # read the number plate
                result = read_license_plate(plate_thresholded)
                if result is not None:
                    license_number, license_number_score = result
                else:
                    license_number, license_number_score = "-1", "-1"

                if license_number != -1 and not None:
                    results[frame_nmb][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                                  'plate': {'bbox': [x1, y1, x2, y2],
                                                            'text': license_number,
                                                            'bbox_score': score,
                                                            'text_score': license_number_score}}
        frame_nmb += 1
    else:
        break

# write results
write_csv(results, './test.csv')
