import face_recognition
import cv2
import numpy as np
import skimage
import skimage.transform
import warnings

def detect_faces(img):
    '''
    Detect faces in image
    :param img: cv::mat HxWx3 RGB
    :return: yield 4 <x,y,w,h>
    '''
    # detect faces
    bbs = face_recognition.face_locations(img)

    for y, right, bottom, x in bbs:
        # Scale back up face bb
        yield x, y, (right - x), (bottom - y)

def detect_biggest_face(img):
    '''
    Detect biggest face in image
    :param img: cv::mat HxWx3 RGB
    :return: 4 <x,y,w,h>
    '''
    # detect faces
    bbs = face_recognition.face_locations(img)

    max_area = float('-inf')
    max_area_i = 0
    for i, (y, right, bottom, x) in enumerate(bbs):
        area = (right - x) * (bottom - y)
        if max_area < area:
            max_area = area
            max_area_i = i

    if max_area != float('-inf'):
        y, right, bottom, x = bbs[max_area_i]
        return x, y, (right - x), (bottom - y)

    return None

def crop_face_with_bb(img, bb):
    '''
    Crop face in image given bb
    :param img: cv::mat HxWx3
    :param bb: 4 (<x,y,w,h>)
    :return: HxWx3
    '''
    x, y, w, h = bb
    return img[y:y+h, x:x+w, :]

def place_face(img, face, bb):
    x, y, w, h = bb
    face = resize_face(face, size=(w, h))
    img[y:y+h, x:x+w] = face
    return img

def resize_face(face_img, size=(128, 128)):
    '''
    Resize face to a given size
    :param face_img: cv::mat HxWx3
    :param size: new H and W (size x size). 128 by default.
    :return: cv::mat size x size x 3
    '''
    return cv2.resize(face_img, size)

def detect_landmarks(face_img):
    landmakrs = face_recognition.face_landmarks(face_img)
    return landmakrs[0] if len(landmakrs) > 0 else None
