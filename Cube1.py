import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO(r'face\best.pt')


#----> To find the Rubick's cube in the image
def yolo_model(image):
    predictions = model.predict(image)
    num_classes = predictions[0].boxes.cls.tolist()
    if(len(num_classes)>0):
       box = predictions[0].boxes.xyxy.tolist()
       # box contains the coordinate for x_min,y_min,x_max,y_max
       x_min,y_min,x_max,y_max = int(box[0][0]),int(box[0][1]),int(box[0][2]),int(box[0][3])
       cropped_image = image[y_min:y_max,x_min:x_max]
       return cropped_image
    else:
       raise Exception("Cube not Detected")


def findCube(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)
    canny = cv.Canny(blurred, 30, 60)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(canny, kernel, iterations=3)
    (contours, _) = cv.findContours(dilated.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    x = []
    y = []
    for contour in contours:
        area = cv.contourArea(contour)
        
        if area < 3000 or area > 20000:
            continue
        
        epsilon = 0.05 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Check if the approximation represents a polygon with sides between 4 and 7
        if len(approx) == 4:
            angles = []
            for i in range(4):
                p0 = approx[i][0]
                p1 = approx[(i + 1) % 4][0]
                p2 = approx[(i + 2) % 4][0]
                vector1 = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                vector2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                dot_product = np.dot(vector1, vector2)
                norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                angle = np.arccos(dot_product / norm_product) * (180 / np.pi)
                angles.append(angle)
            
            if all(80 < angle < 100 for angle in angles):
                min_x = min(approx[:, 0, 0])
                max_x = max(approx[:, 0, 0])
                min_y = min(approx[:, 0, 1])
                max_y = max(approx[:, 0, 1])
                x.append(min_x)
                x.append(max_x)
                y.append(min_y)
                y.append(max_y)
    minix = min(x)
    miniy = min(y)
    maxix = max(x)
    maxiy = max(y)
    crop = image[miniy:maxiy, minix:maxix]
    return crop


# ----> To give color name
def color_name(hsv_value):
    color_ranges = {
        "R": ([0, 70, 50], [10, 255, 255]),
        "R1": ([160, 70, 50], [179, 255, 255]),
        "G": ([40, 40, 40], [80, 255, 255]),
        "B": ([90, 60, 60], [130, 255, 255]),
        "O": ([5, 100, 100], [20, 255, 255]),
        "Y": ([20, 100, 100], [38, 255, 255]),
        "W": ([0, 0, 150], [180, 40, 255]),
    }

    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        if np.all(np.greater_equal(hsv_value, lower_np)) and np.all(np.less_equal(hsv_value, upper_np)):
            if(color == "R1"):
                return "R"
            return color

    return "Unknown"

# ----> To detect a sticker
def face(image, rows=3, cols=3):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    height, width, _ = image.shape
    sticker_height = height // rows
    sticker_width = width // cols
    sticker_colors = []
    for row in range(rows):
        for col in range(cols):
            x1, y1 = col * sticker_width, row * sticker_height
            x2, y2 = (col + 1) * sticker_width, (row + 1) * sticker_height

            x,y = (x1 + x2)//2 , (y1 + y2)//2
            # x,y is approx centre of the sticker

            bgr_value = image[y, x]
            hsv_value = cv.cvtColor(np.uint8([[bgr_value]]), cv.COLOR_BGR2HSV)[0][0]
            sticker_colors.append(hsv_value)

    color_names = [color_name(color) for color in sticker_colors]

    return color_names

def main(img_path):
    image = cv.imread(img_path)
    image = yolo_model(image)
    image = findCube(image)
    colors = face(image)
    ans = []
    ans.append(colors[0:3])
    ans.append(colors[3:6])
    ans.append(colors[6:9])

    return ans

# ------ main ------- #
if __name__ == '__main__':
    ans = main('face\imgs\img93.jpg')
    print(ans)