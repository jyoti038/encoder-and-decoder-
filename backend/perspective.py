import cv2
import numpy as np


class PerspectiveCorrector:

    def __init__(self):
        pass

    # --------------------------------------------------
    # Order Points
    # --------------------------------------------------
    def order_points(self, pts):

        pts = np.array(pts, dtype=np.float32)

        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)

        rect[0] = pts[np.argmin(s)]      # Top Left
        rect[2] = pts[np.argmax(s)]      # Bottom Right

        diff = np.diff(pts, axis=1)

        rect[1] = pts[np.argmin(diff)]   # Top Right
        rect[3] = pts[np.argmax(diff)]   # Bottom Left

        return rect

    # --------------------------------------------------
    # Compute Output Size
    # --------------------------------------------------
    def destination_size(self, rect):

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)

        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)

        maxHeight = int(max(heightA, heightB))

        return maxWidth, maxHeight

    # --------------------------------------------------
    # Perspective Transform
    # --------------------------------------------------
    def warp(self, image, corners):

        rect = self.order_points(corners)

        maxWidth, maxHeight = self.destination_size(rect)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(
            image,
            matrix,
            (maxWidth, maxHeight)
        )

        return warped

    # --------------------------------------------------
    # Perspective Matrix
    # --------------------------------------------------
    def transform_matrix(self, corners):

        rect = self.order_points(corners)

        maxWidth, maxHeight = self.destination_size(rect)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            rect,
            dst
        )

        return matrix

    # --------------------------------------------------
    # Apply Existing Matrix
    # --------------------------------------------------
    def apply(self, image, matrix, width, height):

        return cv2.warpPerspective(
            image,
            matrix,
            (width, height)
        )