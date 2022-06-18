import cv2
import numpy as np

top_right = cv2.imread("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/images/top-right1.jpg", 0)
top_left = cv2.imread("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/images/top-left1.jpg", 0)
bottom_left = cv2.imread("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/images/bottom-left1.jpg", 0)
bottom_right = cv2.imread("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/images/bottom-right1.jpg", 0)

all_corners = [top_left, top_right, bottom_right, bottom_left]


def detect_corners(input_img, corners):

    grey_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    width, height = 2268, 3053
    points = [0,0,0,0,0,0,0,0]

    cl = 0  # corner looper
    for cor in corners:

        w, h = cor.shape[::-1]
        res = cv2.matchTemplate(grey_img, cor, cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(res > threshold)

        # print(loc)
        i = 0

        for pt in zip(*loc[::-1]):
            cv2.rectangle(input_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            print("x1= ", pt[0], "y1= ", pt[1], "x2= ", pt[0]+w, "y2= ",  pt[1] + h)
            if pt[0] > points[cl]:
                points[cl] = pt[0]
            if pt[1] > points[cl+1]:
                points[cl+1] = pt[1]
                # points.append(pt[1])
            i = i + 1

        print("total= ", i, "cl= ", cl)
        cl = cl + 2
    # T-R
    points[2] = points[2]+w

    # B-R
    points[4] = points[4]+w
    points[5] = points[5]+h

    # B-L
    points[7] = points[7] + h

    # draw circles
    j = 0
    while j < 7:
        center1 = points[j], points[j+1]
        cv2.circle(input_img, center1, 5, (0, 0, 255), 3)
        j = j + 2
    crp_img = input_img[points[1]:points[5] ,points[0]:points[4]]
    # cv2.imshow("img", input_img)
    # cv2.imwrite("detected.jpg", input_img)
    # cv2.imwrite("croppped.jpg", crp_img)
    # correct perspective
    #
    x1 = points[0]
    y1 = points[1]

    x2 = points[2]
    y2 = points[3]

    x3 = points[6]
    y3 = points[7]

    x4 = points[4]
    y4 = points[5]


    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    out = cv2.warpPerspective(img, matrix, (width, height))
    grayout = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    (thresh, out_BW) = cv2.threshold(grayout, 150, 255, cv2.THRESH_BINARY)


    # cv2.imwrite("photo_raw_datasets/out1A.jpg", out)
    cv2.imwrite("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/datasets/amhariccrop.jpg", out_BW)
    cropped_center = out_BW[245:2830,245:2120]
    cv2.imwrite("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/datasets/amhariconly.jpg", cropped_center)

    return cropped_center

def shredding(img):
    x = 0
    y = 0
    for i in range(1, 18):
        for j in range(1,14):
            c = img[0+y:150+y ,0+x:150+x] # last was 310 was 60+z

            dim = (32, 32)
            # resize image
            resized = cv2.resize(c, dim, interpolation=cv2.INTER_AREA)
            resized = resized[2:30, 2:30]
            name = "C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/datasets/"+str(i)+"_"+str(j)+ ".jpg"
            cv2.imwrite(name, resized)
            x += 144
        x = 0
        y+= 153

# start of main
img = cv2.imread("C:/Users/guest2/Desktop/last sem/last_project/maraki/maraki_app/static/datasets/dataset.jpg")
out_BW = detect_corners(img, all_corners)
shredding(out_BW)

# c = cropping(cr)
# shredding(img)
# print("four corners: ", cr)




















