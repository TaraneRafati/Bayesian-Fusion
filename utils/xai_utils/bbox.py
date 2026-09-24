import numpy as np
import cv2


def get_bounding_box(mask):
    if np.any(mask):
        mask_uint8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        bboxes = []
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            bboxes.append((x, y, x + w, y + h))
        return bboxes
    return []
