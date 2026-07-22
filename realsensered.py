import cv2
import numpy as np
from camera import RealSenseD435

# HSV 異쒕젰��
hsv = None


def mouse(event, x, y, flags, param):
    global hsv

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel ({x}, {y}) -> HSV = {hsv[y, x]}")


def detect_red_object(camera):

    global hsv

    cv2.namedWindow("Detection")
    cv2.setMouseCallback("Detection", mouse)

    while True:

        color_image, depth_image = camera.get_image()

        if color_image is None:
            continue

        ####################################################
        # ROI (�묒뾽 �곸뿭留� �ъ슜)
        ####################################################
        x1, y1 = 180, 80
        x2, y2 = 1100, 700

        roi_color = color_image[y1:y2, x1:x2]
        roi_depth = depth_image[y1:y2, x1:x2]

        ####################################################
        # HSV
        ####################################################
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 50, 40])
        upper_red1 = np.array([15, 255, 255])

        lower_red2 = np.array([165, 50, 40])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        ####################################################
        # Blur
        ####################################################
        red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)

        ####################################################
        # Morphology
        ####################################################
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=3
        )

        ####################################################
        # 媛��� �� Contour留� �ъ슜
        ####################################################
        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = roi_color.copy()

        has_red = False

        if len(contours) > 0:

            largest = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(largest)

            # 理쒖냼 硫댁쟻
            if area > 400:

                has_red = True

                x, y, w, h = cv2.boundingRect(largest)

                cx = x + w // 2
                cy = y + h // 2

                distance = roi_depth[cy, cx]

                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    result,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )

                text = f"Area:{int(area)}"

                if distance > 0:
                    text += f"  Z:{distance:.3f}m"

                cv2.putText(
                    result,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        ####################################################
        # �곹깭 異쒕젰
        ####################################################
        if has_red:
            color = (0, 255, 0)
            text = "RED : YES"
        else:
            color = (0, 0, 255)
            text = "RED : NO"

        cv2.putText(
            result,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        cv2.imshow("Detection", result)
        cv2.imshow("Mask", red_mask)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    camera = RealSenseD435(
        color_resolution=720,
        depth_mode="720P"
    )

    try:

        detect_red_object(camera)

    finally:

        if hasattr(camera, "_pipeline"):
            camera._pipeline.stop()