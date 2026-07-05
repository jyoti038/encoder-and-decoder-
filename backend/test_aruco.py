import cv2

from aruco import ArucoDetector

image = cv2.imread("aruco_test.jpg")

detector = ArucoDetector()

result = detector.detect(image)

print()

print(result["ids"])

output = detector.draw(

    image,

    result

)

cv2.imwrite(

    "aruco_detected.png",

    output

)

print()

print(

    detector.marker_centers(result)

)