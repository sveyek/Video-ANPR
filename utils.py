import string
import easyocr
import PIL

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

reader = easyocr.Reader(['en'], gpu=False)

# dictionaries for character conversion ambiguities?
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{}, {}, {}, {}, {}, {}, {}\n'.format('frame_nmb', 'car_id', 'car_bbox',
                                                      'plate_bbox', 'plate_bbox_score', 'license_nmb',
                                                      'license_nmb_score'))

        for frame_nmb in results.keys():
            for car_id in results[frame_nmb].keys():
                print(results[frame_nmb][car_id])
                if 'car' in results[frame_nmb][car_id].keys() and \
                        'plate' in results[frame_nmb][car_id].keys() and \
                        'text' in results[frame_nmb][car_id]['plate'].keys():
                    car_bbox = results[frame_nmb][car_id]['car']['bbox']
                    lp = results[frame_nmb][car_id]['plate']
                    f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(frame_nmb,
                                                                  car_id,
                                                                  '[{} {} {} {}]'.format(
                                                                      car_bbox[0],
                                                                      car_bbox[1],
                                                                      car_bbox[2],
                                                                      car_bbox[3]),
                                                                  '[{} {} {} {}]'.format(
                                                                      lp['bbox'][0],
                                                                      lp['bbox'][1],
                                                                      lp['bbox'][2],
                                                                      lp['bbox'][3]),
                                                                  lp['bbox_score'],
                                                                  lp['text'],
                                                                  lp['text_score']
                                                                  ))
        f.close()


def check_license_plate_format(text):
    if len(text) != 7:
        return False

    if (text[0].isupper() or text[0] in dict_int_to_char.keys()) and \
            (text[1].isupper() or text[1] in dict_int_to_char.keys()) and \
            (text[2].isdigit() or text[2] in dict_char_to_int.keys()) and \
            (text[3].isdigit() or text[3] in dict_char_to_int.keys()) and \
            (text[4].isupper() or text[4] in dict_int_to_char.keys()) and \
            (text[5].isupper() or text[5] in dict_int_to_char.keys()) and \
            (text[6].isupper() or text[6] in dict_int_to_char.keys()):
        return True

    else:
        return False


def format_license_number(text):
    license_number = ''

    char_map = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
                2: dict_char_to_int, 3: dict_char_to_int}

    for i in range(7):
        if text[i] in char_map[i].keys():
            license_number += char_map[i][text[i]]
        else:
            license_number += text[i]

    return license_number


def read_license_plate(img):
    detection = reader.readtext(img)

    for det in detection:
        bbox, text, score = det

        text = text.upper().replace(' ', '')

        if check_license_plate_format(text):
            return format_license_number(text), score

        return -1, -1


def map_car(plate, tracking_ids):
    x1, y1, x2, y2, score, class_id = plate

    for j in range(len(tracking_ids)):
        x1car, y1car, x2car, y2car, car_id = tracking_ids[j]

        if x1 > x1car and y1 > y1car and x2 < x2car and y2 < y2car:
            car_index = j
            return tracking_ids[car_index]

    return -1, -1, -1, -1, -1
