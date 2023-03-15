import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = cv2.imread('circulo.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# máscara do contorno do vermelho
image_lower_hsv = np.array([0, 165, 127])
image_upper_hsv = np.array([30, 255, 255])

mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
mask_hsv = cv2.erode(mask_hsv, None, iterations=2)
mask_hsv = cv2.dilate(mask_hsv, None, iterations=2)

# azul
image_lower_hsv2 = np.array([80, 165, 127])
image_upper_hsv2 = np.array([90, 230, 230])

mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)
mask_hsv2 = cv2.erode(mask_hsv2, None, iterations=2)
mask_hsv2 = cv2.dilate(mask_hsv2, None, iterations=2)

# somamos as duas mascaras em uma imagem
mask_total = cv2.bitwise_or(mask_hsv, mask_hsv2)

mask_rgb = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2RGB)
contornos_img = mask_rgb.copy()
contornos, _ = cv2.findContours(
    mask_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

centros = []

for c in contornos:
    size = 20
    color = (128, 128, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    M = cv2.moments(c)

    if (M['m00'] != 0):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        centro = [cx, cy]
        centros.append(centro)

        c_area = cv2.contourArea(c)
        cv2.putText(contornos_img, str(
            f'area: {c_area}'), (cx - 80, cy + 120), font, 0.6, (200, 50, 0), 2, cv2.LINE_AA)
        cv2.putText(contornos_img, str(
            f'centro: [{cx},{cy}]'), (cx - 80, cy - 100), font, 0.6, (200, 50, 0), 2, cv2.LINE_AA)

        cv2.line(contornos_img, (cx - size, cy), (cx + size, cy), color, 5)
        cv2.line(contornos_img, (cx, cy - size), (cx, cy + size), color, 5)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5)


# desenha linha entre centros
if (len(centros) > 0):
    cv2.line(contornos_img, centros[0], centros[1], color, 5)

    p2 = centros[0]
    p1 = centros[1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)
    if angle < 0:
        angle = 360 + angle

    cv2.putText(contornos_img, str(
        f'angulo: {angle}'), (10, 520), font, 0.6, (200, 50, 0), 2, cv2.LINE_AA)

plt.imshow(contornos_img)
plt.show()
