import numpy as np
import cv2


def extract_intracranial_mask(img_hu: np.ndarray) -> np.ndarray:

    _, thresh = cv2.threshold(img_hu.astype(np.float32), 0, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)

    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    solid_head = np.zeros_like(thresh)
    if contours:
        for i in range(len(contours)):
            cv2.drawContours(solid_head, contours, i, 255, -1)

    skull_bone = (img_hu > 100).astype(np.uint8) * 255

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    expanded_skull = cv2.dilate(skull_bone, kernel_dilate)

    inner_brain_space = cv2.bitwise_and(solid_head, cv2.bitwise_not(expanded_skull))

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(inner_brain_space, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    final_intracranial_mask = cv2.erode(cleaned_mask, kernel_clean, iterations=3)

    return (final_intracranial_mask > 0).astype(np.uint8)
