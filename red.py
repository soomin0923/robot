import cv2
import numpy as np


def detect_red_object(
    image,
    min_red_area=1000,
    show_result=True
):
    """
    �대�吏��� �쇱젙 �ш린 �댁긽�� 鍮④컙�� �곸뿭�� 議댁옱�섎뒗吏� �먮떒�⑸땲��.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV濡� �쎌� BGR �대�吏�
    min_red_area : int
        鍮④컙�됱쑝濡� �먮떒�� 理쒖냼 �쎌� 硫댁쟻
    show_result : bool
        寃곌낵 李� 異쒕젰 �щ�

    Returns
    -------
    has_red : bool
        鍮④컙�� 臾쇱껜 議댁옱 �щ�
    red_area : int
        異붿텧�� 鍮④컙�� �곸뿭�� �꾩껜 �쎌� ��
    mask : numpy.ndarray
        鍮④컙�� �곸뿭 留덉뒪��
    """

    if image is None:
        raise ValueError("�대�吏�瑜� 遺덈윭�ㅼ� 紐삵뻽�듬땲��.")

    # BGR �� HSV 蹂���
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 鍮④컙�� HSV 踰붿쐞 1
    lower_red1 = np.array([0, 80, 60], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # 鍮④컙�� HSV 踰붿쐞 2
    lower_red2 = np.array([170, 80, 60], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    # �� 鍮④컙�� 踰붿쐞�� ���� 留덉뒪�� �앹꽦
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # �묒� �몄씠利� �쒓굅
    kernel = np.ones((5, 5), dtype=np.uint8)

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # �딆뼱吏� 鍮④컙�� �곸뿭 �곌껐
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # 鍮④컙�� �쎌� �� 怨꾩궛
    red_area = cv2.countNonZero(red_mask)

    # 理쒖냼 硫댁쟻 �댁긽�대㈃ 鍮④컙�� 臾쇱껜媛� 議댁옱�쒕떎怨� �먮떒
    has_red = red_area >= min_red_area

    if show_result:
        result = image.copy()

        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # �덈Т �묒� 鍮④컙�� �몄씠利덈뒗 �쒖쇅
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                result,
                f"Red area: {int(area)}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        status_text = "RED OBJECT: YES" if has_red else "RED OBJECT: NO"
        status_color = (0, 255, 0) if has_red else (0, 0, 255)

        cv2.putText(
            result,
            status_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            status_color,
            3
        )

        # �먮낯 �대�吏�媛� �щ�濡� �붾㈃ 異쒕젰�⑹쑝濡� 異뺤냼
        display_width = 1000
        scale = display_width / result.shape[1]
        display_height = int(result.shape[0] * scale)

        result_display = cv2.resize(
            result,
            (display_width, display_height)
        )

        mask_display = cv2.resize(
            red_mask,
            (display_width, display_height)
        )

        cv2.imshow("Detection Result", result_display)
        cv2.imshow("Red Mask", mask_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return has_red, red_area, red_mask


if __name__ == "__main__":
    image = cv2.imread("/home/js/ros2_ws/src/doosan-robot2/test_img.jpg")

    has_red, red_area, mask = detect_red_object(
        image,
        min_red_area=1000
    )
    print(f"has_red?: {has_red}")

    print(f"鍮④컙�� �쎌� ��: {red_area}")

    if has_red:
        print("�먯젙 寃곌낵: 鍮④컙�됱씠 �덈뒗 臾쇱껜媛� 議댁옱�⑸땲��.")
    else:
        print("�먯젙 寃곌낵: 鍮④컙�됱씠 �덈뒗 臾쇱껜媛� �놁뒿�덈떎.")



