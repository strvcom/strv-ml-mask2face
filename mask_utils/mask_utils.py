# Author: aqeelanwar
# Created: 27 April,2020, 10:21 PM
# Email: aqeel.anwar@gatech.edu
# github: https://github.com/aqeelanwar/MaskTheFace

import copy
import cv2
import math
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageColor, ImageFilter
from configparser import ConfigParser

from mask_utils.fit_ellipse import *
from mask_utils.read_cfg import read_cfg

COLOR = [
    "#fc1c1a",
    "#177ABC",
    "#94B6D2",
    "#A5AB81",
    "#DD8047",
    "#6b425e",
    "#e26d5a",
    "#c92c48",
    "#6a506d",
    "#ffc900",
    "#ffffff",
    "#000000",
    "#49ff00",
]


def get_line(face_landmark, pil_image, type="eye", debug=False):
    d = ImageDraw.Draw(pil_image)
    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (
                face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2]
        # mid_point = (
        #     face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        # ) / 2

        mid_pointY = (
                             face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
                     ) / 2
        mid_pointX = (
                             face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]
                     ) / 2
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (
                face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (
                            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
                    ) / 2

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = eye_line_mid
        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    # d.line(eye_mid, width=5, fill='red')
    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]
    # cv2.imshow('h', image)
    # cv2.waitKey(0)
    eye_line = fit_line(x, y, pil_image)
    d.line(eye_line, width=5, fill="blue")

    # Perpendicular Line
    # (midX, midY) and (midX - y2 + y1, midY + x2 - x1)
    y = [
        (left_point[1] + right_point[1]) / 2,
        (left_point[1] + right_point[1]) / 2 + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2,
        (left_point[0] + right_point[0]) / 2 - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, pil_image)
    if debug:
        d.line(perp_line, width=5, fill="red")
        pil_image.show()
    return eye_line, perp_line, left_point, right_point, mid_point


def get_points_on_chin(line, face_landmark, chin_type="chin"):
    chin = face_landmark[chin_type]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    return points_on_chin


def plot_lines(face_line, image, debug=False):
    pil_image = Image.fromarray(image)
    if debug:
        d = ImageDraw.Draw(pil_image)
        d.line(face_line, width=4, fill="white")
        pil_image.show()


def line_intersection(line1, line2):
    # mid = int(len(line1) / 2)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
            segment_maxX + 1 >= x >= segment_minX - 1
            and segment_maxY + 1 >= y >= segment_minY - 1
    ):
        flag = True

    return flag, x, y


def fit_line(x, y, image):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image.size[1], 50)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_six_points(face_landmark, image):
    _, perp_line1, _, _, m = get_line(face_landmark, image, type="nose_mid")
    face_b = m

    perp_line, _, _, _, _ = get_line(face_landmark, image, type="perp_line")
    points1 = get_points_on_chin(perp_line1, face_landmark)
    points = get_points_on_chin(perp_line, face_landmark)
    if not points1:
        face_e = tuple(np.asarray(points[0]))
    elif not points:
        face_e = tuple(np.asarray(points1[0]))
    else:
        face_e = tuple((np.asarray(points[0]) + np.asarray(points1[0])) / 2)
    # face_e = points1[0]
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_long")

    angle = get_angle(perp_line, nose_mid_line)
    # print("angle: ", angle)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_tip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    if len(points) < 2:
        face_landmark = get_face_ellipse(face_landmark)
        # print("extrapolating chin")
        points = get_points_on_chin(
            nose_mid_line, face_landmark, chin_type="chin_extrapolated"
        )
        if len(points) < 2:
            points = []
            points.append(face_landmark["chin"][0])
            points.append(face_landmark["chin"][-1])
    face_a = points[0]
    face_c = points[-1]
    # cv2.imshow('j', image)
    # cv2.waitKey(0)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="bottom_lip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    face_d = points[0]
    face_f = points[-1]

    six_points = np.float32([face_a, face_b, face_c, face_f, face_e, face_d])

    return six_points, angle


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    # print("perp", perp_angle)
    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180
    # print("nose", nose_angle)

    angle = nose_angle - perp_angle
    return angle


def mask_image(image, face_location, configuration):
    mask_type = configuration.get('mask_type')
    if mask_type == "random":
        available_mask_types = get_available_mask_types()
        mask_type = random.choice(available_mask_types)

    x = [99999, 0]
    y = [99999, 0]
    for point in face_location:
        if point[0] < x[0]:
            x[0] = point[0]
        if point[0] > x[1]:
            x[1] = point[0]
        if point[1] < y[0]:
            y[0] = point[1]
        if point[1] > y[0]:
            y[1] = point[1]

    face_landmarks = shape_to_landmarks(face_location)
    six_points, angle = get_six_points(face_landmarks, image.copy())

    # Find the face angle
    threshold = 13
    if angle < -threshold:
        mask_type += "_right"
    elif angle > threshold:
        mask_type += "_left"

    cfg = read_cfg(config_filename="masks/masks.cfg", mask_type=mask_type, verbose=False)
    img = cv2.imread(cfg.template, cv2.IMREAD_UNCHANGED)

    # Process the mask if necessary
    if configuration.get('mask_patter'):
        # Apply pattern to mask
        img = texture_the_mask(img, configuration.get('mask_patter'), configuration.get('mask_pattern_weight'))

    if configuration.get('mask_color'):
        # Apply color to mask
        img = color_the_mask(img, configuration.get('mask_color'), configuration.get('mask_color_weight'))

    mask_line = np.float32(
        [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
    )

    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (image.size[0], image.size[1]), flags=cv2.INTER_CUBIC)

    img_cv = cv2.cvtColor(dst_mask, cv2.COLOR_BGRA2RGBA)
    f = Image.fromarray(img_cv, 'RGBA')

    mask = np.array(dst_mask)
    mask = np.clip(np.sum(mask, axis=2), 0, 255)
    mask_img = Image.fromarray(mask.astype('uint8'), 'L')
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=9))
    masked_face = Image.composite(f, image.convert('RGBA'), mask_img)
    if configuration.get('mask_filter_output'):
        masked_face = masked_face.filter(ImageFilter.GaussianBlur(radius=configuration.get('mask_filter_radius')))
    return masked_face


def draw_landmarks(face_landmarks, image):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="white")
    pil_image.show()


def get_face_ellipse(face_landmark):
    chin = face_landmark["chin"]
    x = []
    y = []
    for point in chin:
        x.append(point[0])
        y.append(point[1])

    x = np.asarray(x)
    y = np.asarray(y)

    a = fitEllipse(x, y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    a, b = axes

    arc = 2.2
    R = np.arange(0, arc * np.pi, 0.2)
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    chin_extrapolated = []
    for i in range(len(R)):
        chin_extrapolated.append((xx[i], yy[i]))
    face_landmark["chin_extrapolated"] = chin_extrapolated
    return face_landmark


def shape_to_landmarks(shape):
    face_landmarks = {}
    face_landmarks["left_eyebrow"] = [
        tuple(shape[17]),
        tuple(shape[18]),
        tuple(shape[19]),
        tuple(shape[20]),
        tuple(shape[21]),
    ]
    face_landmarks["right_eyebrow"] = [
        tuple(shape[22]),
        tuple(shape[23]),
        tuple(shape[24]),
        tuple(shape[25]),
        tuple(shape[26]),
    ]
    face_landmarks["nose_bridge"] = [
        tuple(shape[27]),
        tuple(shape[28]),
        tuple(shape[29]),
        tuple(shape[30]),
    ]
    face_landmarks["nose_tip"] = [
        tuple(shape[31]),
        tuple(shape[32]),
        tuple(shape[33]),
        tuple(shape[34]),
        tuple(shape[35]),
    ]
    face_landmarks["left_eye"] = [
        tuple(shape[36]),
        tuple(shape[37]),
        tuple(shape[38]),
        tuple(shape[39]),
        tuple(shape[40]),
        tuple(shape[41]),
    ]
    face_landmarks["right_eye"] = [
        tuple(shape[42]),
        tuple(shape[43]),
        tuple(shape[44]),
        tuple(shape[45]),
        tuple(shape[46]),
        tuple(shape[47]),
    ]
    face_landmarks["top_lip"] = [
        tuple(shape[48]),
        tuple(shape[49]),
        tuple(shape[50]),
        tuple(shape[51]),
        tuple(shape[52]),
        tuple(shape[53]),
        tuple(shape[54]),
        tuple(shape[60]),
        tuple(shape[61]),
        tuple(shape[62]),
        tuple(shape[63]),
        tuple(shape[64]),
    ]

    face_landmarks["bottom_lip"] = [
        tuple(shape[54]),
        tuple(shape[55]),
        tuple(shape[56]),
        tuple(shape[57]),
        tuple(shape[58]),
        tuple(shape[59]),
        tuple(shape[48]),
        tuple(shape[64]),
        tuple(shape[65]),
        tuple(shape[66]),
        tuple(shape[67]),
        tuple(shape[60]),
    ]

    face_landmarks["chin"] = [
        tuple(shape[0]),
        tuple(shape[1]),
        tuple(shape[2]),
        tuple(shape[3]),
        tuple(shape[4]),
        tuple(shape[5]),
        tuple(shape[6]),
        tuple(shape[7]),
        tuple(shape[8]),
        tuple(shape[9]),
        tuple(shape[10]),
        tuple(shape[11]),
        tuple(shape[12]),
        tuple(shape[13]),
        tuple(shape[14]),
        tuple(shape[15]),
        tuple(shape[16]),
    ]
    return face_landmarks


def get_available_mask_types(config_filename="masks/masks.cfg"):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    available_mask_types = parser.sections()
    available_mask_types = [
        string for string in available_mask_types if "left" not in string
    ]
    available_mask_types = [
        string for string in available_mask_types if "right" not in string
    ]

    return available_mask_types


def color_the_mask(mask_image, color, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    RGB_color = ImageColor.getcolor(color, "RGB")
    RGB_color = (RGB_color[2], RGB_color[1], RGB_color[0])
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]

    color_image = np.full(mask_image.shape, RGB_color, np.uint8)
    mask_color = cv2.addWeighted(mask_image, 1 - intensity, color_image, intensity, 0)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=bit_mask)
    colored_mask = np.zeros(orig_shape, dtype=np.uint8)
    colored_mask[:, :, 0:3] = mask_color
    colored_mask[:, :, 3] = bit_mask
    return colored_mask


def texture_the_mask(mask_image, texture_path, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]
    texture_image = cv2.imread(texture_path)
    texture_image = cv2.resize(texture_image, (orig_shape[1], orig_shape[0]))

    mask_texture = cv2.addWeighted(
        mask_image, 1 - intensity, texture_image, intensity, 0
    )
    mask_texture = cv2.bitwise_and(mask_texture, mask_texture, mask=bit_mask)
    textured_mask = np.zeros(orig_shape, dtype=np.uint8)
    textured_mask[:, :, 0:3] = mask_texture
    textured_mask[:, :, 3] = bit_mask

    return textured_mask
