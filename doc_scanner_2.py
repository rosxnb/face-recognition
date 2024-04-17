import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from hed.hed import detect_edge


def img_threshold(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gw, gs, gw1, gs1, gw2, gs2 = (3, 1.0, 7, 3.0, 3, 2.0)
    img_blur = cv.GaussianBlur(img_gray, (gw, gw), gs)
    g1 = cv.GaussianBlur(img_blur, (gw1, gw1), gs1)
    g2 = cv.GaussianBlur(img_blur, (gw2, gw2), gs2)

    ret, thres = cv.threshold(g2-g1, 127, 255, cv.THRESH_BINARY)
    return ret, thres


def doc_scanner():
    # path = "./data/copy_img.jpeg"
    path = "./data/docs/data/citizenship_back/179815901_4066583826740498_7102551088841188490_n.jpg"
    img = cv.imread(path)
    img_copy = img.copy()

    # _, thresholds = img_threshold(img)
    _, thresholds = detect_edge(img)
    print(f"{thresholds.shape = }")
    # thresholds = cv.erode(img, np.ones((5, 5)), iterations=2)

    plt.imshow(thresholds, cmap="gray")
    plt.show()
    contours, _ = cv.findContours(thresholds, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    width, height = 0, 0
    start_x, end_x = 0, 0
    start_y, end_y = 0, 0

    for idx, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        area = w * h
        # asp_ratio = float(w)/h

        if area > 1_000:
            approx = cv.approxPolyDP(
                contour,
                0.05 * cv.arcLength(contour, False),
                False
            )

            # if len(approx) == 4:
            width = w
            height = h
            start_x = x
            start_y = y
            end_x = start_x + width
            end_y = start_y + height
            cv.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (255, 0, 0), 10)
            cv.putText(img_copy, "Document " + str(x) + ", " + str(y), (x, y-5), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0) )
            # else: print(f"{idx = }, {len(approx) = }")
            print(f"{idx = },  {area = }")
            print(f"{idx = }, {len(approx) = }")

    plt.imshow(img_copy)
    plt.show()
    print(f"Start coords: {start_x, start_y}")
    print(f"End coords: {end_x, end_y}")

doc_scanner()
