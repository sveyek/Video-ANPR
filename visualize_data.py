import ast
import cv2 as cv
import numpy as np
import pandas as pd

# read interpolated data
res = pd.read_csv('./test_interpolated.csv')

cap = cv.VideoCapture('demo_1.mp4')

# load video to write
fourcc = cv.VideoWriter.fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
output = cv.VideoWriter('./output.mp4', fourcc, fps, (width, height))

license_plate = {}

# identify the number plate with most confidence score for each car
for car_id in np.unique(res['car_id']):
    idx = res[(res['car_id'] == car_id)]['license_nmb_score'].idxmax()
    license_number = res.loc[idx, 'license_nmb']
    frame_number = res.loc[idx, 'frame_nmb']
    plate_bbox = res.loc[idx, 'plate_bbox']

    license_number.replace(' ','')

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(plate_bbox.replace(' ', ','))

    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    plate_crop = cv.resize(plate_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id] = {
        'license_crop': plate_crop,
        'license_plate_nmb': license_number
    }

frame_nmb = 0

cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    print(frame_nmb)
    if ret:
        df = res[res['frame_nmb'] == frame_nmb]
        for row_idx in range(len(df)):
            # draw car
            x1car, y1car, x2car, y2car = map(int, ast.literal_eval(df.iloc[row_idx]['car_bbox'].replace(' ', ',')))
            cv.rectangle(frame, (x1car, y1car), (x2car, y2car), (0, 255, 0), 15)

            # draw license plate
            x1, y1, x2, y2 = map(int, ast.literal_eval(df.iloc[row_idx]['plate_bbox'].replace(' ', ',')))
            # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            license_crop = license_plate[df.iloc[row_idx]['car_id']]['license_crop']
            H, W, _ = license_crop.shape

            try:
                # number plate
                text = license_plate[df.iloc[row_idx]['car_id']]['license_plate_nmb']

                # if not detected
                if text in ['0', '-1']:
                    continue

                # text size calc
                (text_width, text_height), baseline = cv.getTextSize(
                    text,
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    10)

                # white patch
                frame[y1car - 50 - text_height:y1car - 10, x1car: x1car + text_width, :] = (255, 255, 255)

                # put text
                cv.putText(
                    frame,
                    text,
                    (x1car, y1car - 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 0),
                    10)
            except:
                pass

        output.write(frame)
        frame = cv.resize(frame, (1280, 729))

    frame_nmb += 1

output.release()
cap.release()
