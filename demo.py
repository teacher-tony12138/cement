import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Please input image path')
parser.add_argument('img_index', type=str)
args = parser.parse_args()

img_path = './imgs'
img_list = ['32_1.jpeg', '32_2.jpeg', '32_3.jpeg', '32_4.jpeg']

img_file = f'{img_path}/32_{args.img_index}.jpeg'

img = cv2.imread(img_file)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 70, 210)

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
res = cv2.drawContours(img, contours,-1,(0,255,0),2)

for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screen_cnt = approx
		break

def crop_img(cord, target):
    x = [p[0][0] for p in cord]
    y = [p[0][1] for p in cord]
    print(f'Image size is {target.shape[0], target.shape[1]}')
    print(f'Cordinates are:')
    for idx, p in enumerate(cord):
        print(f'{idx}: {p[0]}')
    return target[min(y): max(y), min(x): max(x), :]

croped = crop_img(screen_cnt, img)

def detect_angle(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adaptive = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,4)

    cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 45000 and area > 20:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = mask.shape
    
    # Horizontal
    if w > h:
        left = mask[0:h, 0:0+w//2]
        right = mask[0:h, w//2:]
        left_pixels = cv2.countNonZero(left)
        right_pixels = cv2.countNonZero(right)
        return 0 if left_pixels >= right_pixels else 180
    # Vertical
    else:
        top = mask[0:h//2, 0:w]
        bottom = mask[h//2:, 0:w]
        top_pixels = cv2.countNonZero(top)
        bottom_pixels = cv2.countNonZero(bottom)
        return 90 if bottom_pixels >= top_pixels else 270

angle = detect_angle(croped)
print(f'Angle is {angle}')


while True:
    cv2.imshow('result', res)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()


