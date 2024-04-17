import cv2 as cv
import numpy as np
from stack_images import stack_images
from hed.hed import detect_edge


img_width = 540
img_height = 680


def draw_circles(img, points, radius=5, color=(0, 0, 0)):
    for x, y in points:
        cv.circle(
            img,
            center=(x, y),
            radius=radius,
            color=color,
            thickness=cv.FILLED,
        )


def get_contours(img, threshold=3000):
    result = np.array([])
    max_area = 0

    contours, _ = cv.findContours(
        image=img,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_NONE,
    )

    for contour in contours:
        area = cv.contourArea(contour)

        if area > threshold:
            peri = cv.arcLength(curve=contour, closed=True)
            edges = cv.approxPolyDP(curve=contour, epsilon=0.02*peri, closed=True)
            # print(f"{len(edges) = }")

            # if len(edges) == 5:
            #     x, y, w, h = cv.boundingRect(array=edges)
            #     cv.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 3)
            #     # print(f"{edges.shape = }")
            #     draw_circles(img_contours, edges.squeeze())
            #     result = edges

            if len(edges) == 4 and area > max_area:
                # print("len == 4")
                result = edges
                max_area = area
                print(f"{max_area = }")

    cv.drawContours(image=img, contours=result, contourIdx=-1, color=(255, 0, 0), thickness=5)
    return result


def reorder(points):
    points = points.squeeze()
    points_sum = points.sum(axis=1)
    new_points = np.zeros( points.shape, np.int32 )

    new_points[0] = points[ np.argmin(points_sum) ]
    new_points[3] = points[ np.argmax(points_sum) ]

    diff = np.diff(points, axis=1)
    new_points[1] = points[ np.argmin(diff) ]
    new_points[2] = points[ np.argmax(diff) ]

    return new_points


def get_warp(img, corners):
    corners = reorder(corners)
    pts1 = np.float32(corners)
    pts2 = np.float32([
        [0, 0],
        [img_width, 0],
        [0, img_height],
        [img_width, img_height],
    ])

    perspective_mat = cv.getPerspectiveTransform(
        src=pts1,
        dst=pts2,
    )

    img_output = cv.warpPerspective(
        src=img,
        M=perspective_mat,
        dsize=(img_width, img_height),
    )

    return img_output


def static_doc_scanner():
    img = cv.imread("./data/docscanner.jpg")
    # img = cv.resize(img, (img_width, img_height))
    img_contour = img.copy()

    # img_threshold = pre_process(img)
    _, img_threshold = detect_edge(img)
    img_contours = get_contours(img_threshold, img_contour)

    cv.imshow("Original", img_threshold)
    # cv.imshow("Processed", img_threshold)
    cv.imshow("Contours", img_contour)
    # cv.imshow("Stacked", ([img, img_threshold, img_contours]))

    _ = reorder(img_contours)

    key = cv.waitKey(0)
    if key == ord("q"):
        print("gracefull exit")

    cv.destroyAllWindows()


def real_time_scanner():
    # cam = cv.VideoCapture(1)
    cam = cv.VideoCapture(0)
    black_img = np.zeros( (img_width, img_height), np.uint8 )
    count: int = 0

    while True:
        _, img = cam.read()
        print(f"{img.shape = }")
        if img is None:
            img = black_img

        # img_threshold = pre_process(img)
        _, img_threshold = detect_edge(img)
        detected_obj = get_contours(img_threshold, threshold=5_000)

        if detected_obj.size != 0:
            img_warpped = get_warp(img, detected_obj)
            img_stack = ([img_threshold, img_warpped])
            count += 1
            print(f"{count = }")
            # cv.imshow("ImageWarped", img_warpped)

        else:
            img_stack = ([img, img_threshold])

        stacked_images = stack_images(1, img_stack)
        cv.imshow("Images", stacked_images)

        key = cv.waitKey(40)
        if key == ord("q"):
            print("gracefull exit")
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    # static_doc_scanner()
    real_time_scanner()
