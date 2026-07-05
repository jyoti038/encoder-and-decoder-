import cv2

from perspective import PerspectiveCorrector

image = cv2.imread("phone_capture.jpg")

corrector = PerspectiveCorrector()

# Example corner points
corners = [

    [120, 90],

    [930, 110],

    [920, 980],

    [110, 960]

]

corrected = corrector.warp(

    image,

    corners

)

cv2.imwrite(

    "corrected.png",

    corrected

)

print("Perspective correction completed.")