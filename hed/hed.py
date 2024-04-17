"""
    Original HED paper: https://arxiv.org/pdf/1504.06375.pdf
    HED - Hollistically-Nested Edge Detection

    Caffe model is encoded into two files
        1. Proto text file: https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt
        2. Pretrained caffe model: http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel
"""

import numpy as np
import cv2 as cv
from cv2 import dnn



proto_path = "hed/deploy.prototxt"
model_path = "hed/hed_pretrained_bsds.caffemodel"
model = dnn.readNetFromCaffe(proto_path, model_path)


# def _load_model():
#     proto_path = "hed/deploy.prototxt"
#     model_path = "hed/hed_pretrained_bsds.caffemodel"
#     model = dnn.readNetFromCaffe(proto_path, model_path)
#
#     def get_model():
#         return model
#
#     return get_model


def get_image_blob(img):
    (H, W) = img.shape[:2]
    mean_pixel_values = np.average(img, axis=(0, 1))

    blob = dnn.blobFromImage(
        img,
        scalefactor=0.7,
        size=(W, H),
        mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
        swapRB=False,
        crop=False,
    )

    return blob


def detect_edge(img):
    blob = get_image_blob(img)
    # model = _load_model()()

    model.setInput(blob)
    y_hat = model.forward()
    y_hat = np.squeeze(y_hat) # y_hat = y_hat[0, 0, :, :] # select only last two axis
    y_hat = (y_hat * 255).astype("uint8")

    return blob, y_hat


def _main():
    path = "data/docscanner.jpg"
    # path = "../data/document1.jpg"

    img = cv.imread(path)
    return detect_edge(img)


if __name__ == "__main__":
    blob, y_hat = _main()
    visualizable_blob = np.transpose( blob.squeeze(), (1, 2, 0) )
