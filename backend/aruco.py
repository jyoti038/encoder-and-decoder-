import cv2
import numpy as np


class ArucoDetector:

    def __init__(self):

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.parameters
        )

    # --------------------------------------------------
    # Detect ArUco Markers
    # --------------------------------------------------

    def detect(self, image):

        corners, ids, rejected = self.detector.detectMarkers(image)

        return {

            "corners": corners,

            "ids": ids,

            "rejected": rejected

        }

    # --------------------------------------------------
    # Marker Found?
    # --------------------------------------------------

    def has_markers(self, detection):

        ids = detection["ids"]

        return ids is not None and len(ids) > 0

    # --------------------------------------------------
    # Draw Markers
    # --------------------------------------------------

    def draw(self, image, detection):

        output = image.copy()

        if self.has_markers(detection):

            cv2.aruco.drawDetectedMarkers(

                output,

                detection["corners"],

                detection["ids"]

            )

        return output

    # --------------------------------------------------
    # Get Marker Centers
    # --------------------------------------------------

    def marker_centers(self, detection):

        centers = []

        if not self.has_markers(detection):

            return centers

        for marker in detection["corners"]:

            pts = marker[0]

            cx = np.mean(pts[:, 0])

            cy = np.mean(pts[:, 1])

            centers.append((cx, cy))

        return centers

    # --------------------------------------------------
    # Sort Markers by ID
    # --------------------------------------------------

    def sort_by_id(self, detection):

        if not self.has_markers(detection):

            return []

        markers = []

        for marker_id, corner in zip(

            detection["ids"],

            detection["corners"]

        ):

            markers.append({

                "id": int(marker_id[0]),

                "corner": corner

            })

        markers.sort(

            key=lambda x: x["id"]

        )

        return markers