import cv2

# This class transformes the image to the corrected perspective with the warp functions
class PerTransformerClass:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        # print(self.M)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        # print(self.M_inv)

    def transform(self, img):
        # print("I used transform")
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        # print("I used inverse transform")
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
