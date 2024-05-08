import cv2

class transforms:
    def __init__(self):
        pass

    def rotate_scale(self, image, angle, scale=1.0):
        center_X, center_Y = (image.shape[:2][1] // 2, image.shape[:2][0] // 2)
        rot_mat = cv2.getRotationMatrix2D((center_X, center_Y), angle, scale)

        rot_image = cv2.warpAffine(image, rot_mat, (image.shape[:2][1], image.shape[:2][0]))

        return rot_image