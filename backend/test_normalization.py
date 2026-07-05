import cv2

from normalization import ImageNormalizer

image = cv2.imread("phone_capture.jpg")

normalizer = ImageNormalizer()

result = normalizer.normalize(image)

cv2.imwrite(
    "normalized.png",
    result
)

print("Normalization Completed.")