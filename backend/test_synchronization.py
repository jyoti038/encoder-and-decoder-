import cv2

from synchronization import Synchronizer

image = cv2.imread("normalized.png")

sync = Synchronizer()

aligned = sync.synchronize(
    image,
    (1024, 1024)
)

cv2.imwrite(
    "aligned.png",
    aligned
)

print(sync.info(aligned))