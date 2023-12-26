import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_for_missing_frames(data):
    frame_numbers = np.array([int(row['frame_nmb']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][2:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['plate_bbox'][2:-1].split())) for row in data])

    interpolated_data = []
    car_id_set = np.unique(car_ids)

    for car in car_id_set:
        frames = [p['frame_nmb'] for p in data if int(float(p['car_id'])) == int(float(car))]  # can remove float?

        mask = car_ids == car
        frames_masked = frame_numbers[mask]
        car_bboxes_masked = car_bboxes[mask]
        plates_masked = license_plate_bboxes[mask]

        car_bboxes_interpolated = []
        plate_bboxes_interpolated = []

        first_frame = frames_masked[0]
        last_frame = frames_masked[-1]

        for i in range(len(frames_masked)):  # check again
            frame = frames_masked[i]
            car_bbox = car_bboxes_masked[i]
            plate_bbox = plates_masked[i]

            if i > 0:  # previous frame exists?
                prev_frame = frames_masked[i - 1]
                prev_car_bbox = car_bboxes_masked[i - 1]
                prev_plate_bbox = plates_masked[i - 1]

                if frame - prev_frame > 1:  # are there any missing frames?
                    frame_gap = frame - prev_frame
                    x = np.array([prev_frame, frame])
                    x_new = np.linspace(prev_frame, frame, num=frame_gap, endpoint=False)
                    interpolator_car = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interpolator_car(x_new)
                    interpolator_plate = interp1d(x, np.vstack((prev_plate_bbox, plate_bbox)), axis=0, kind='linear')
                    interpolated_plate_bboxes = interpolator_plate(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes)
                    plate_bboxes_interpolated.extend(interpolated_plate_bboxes)

            car_bboxes_interpolated.append(car_bbox)
            plate_bboxes_interpolated.append(plate_bbox)

        for j in range(len(car_bboxes_interpolated)):
            frame_nmb = first_frame + j
            row = {'frame_nmb': frame_nmb, 'car_id': str(car),
                   'car_bbox': ' '.join(map(str, car_bboxes_interpolated[j])),
                   'plate_bbox': ' '.join(map(str, plate_bboxes_interpolated[j]))}

            if str(frame_nmb) not in frames:
                row['plate_bbox_score'] = '0'
                row['license_nmb'] = '0'
                row['license_nmb_score'] = '0'
            else:
                original_row = [p for p in data if
                                int(p['frame_nmb']) == frame_nmb and int(float(p['car_id'])) == int(float(car))][0]
                row['plate_bbox_score'] = original_row[
                    'plate_bbox_score'] if 'plate_bbox_score' in original_row else '0'
                row['license_nmb'] = original_row['license_nmb'] if 'license_nmb' in original_row else '0'
                row['license_nmb_score'] = original_row[
                    'license_nmb_score'] if 'license_nmb_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


with open('test.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

interpolated = interpolate_for_missing_frames(data)

header = ['frame_nmb', 'car_id', 'car_bbox', 'plate_bbox', 'plate_bbox_score', 'license_nmb', 'license_nmb_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated)
