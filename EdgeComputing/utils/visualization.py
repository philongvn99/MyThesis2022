"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""


import numpy as np
import cv2


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

maxOVL = 0.9
minOVL = 0.35

def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs
  
#===============  DETECT MOTIon OBJECTS ===============================#   


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img

#================================== CHECK OVERLAP BOX EXIST ==========================#  
def isOverlap(new, old):
    dx = min(old[2], new[2]) - max(old[0], new[0])
    dy = min(old[3], new[3]) - max(old[1], new[1])
    sx = new[2] - new[0]
    sy = new[3] - new[1]
    return ((dx*dy) / (sx*sy)) > minOVL if (dx >= 0) and (dy >= 0) else False

#-------- Check if any overlap Bounding Box in Last Frame -------#
def isOverlapExist(box, frameBoxes):
    return False if frameBoxes == [] else any([isOverlap(box, fb) for fb in frameBoxes])




class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, cls_dict, bg):
        self.cls_dict = cls_dict
        self.colors = [(0., 0., 0.), (255., 0., 0.), (0., 255., 0.), (0., 0., 255.)]  #gen_colors(len(cls_dict))
        self.frameList = []
        self.background = cv2.cvtColor(bg, cv2.COLOR_BAYER_BG2GRAY)
        
    
    def findCountours(self, img):
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, self.background)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=10)
        # append the final result into the `frame_diff_list`
        self.background = gray
        # self.frameList.append(dilate_frame)
        # # if we have reached `consecutive_frame` number of frames
        # if len(self.frameList) == 3:
        #     # add all the frames in the `frame_diff_list`
        #     sum_frames = sum(self.frameList)
        #     cv2.imshow('aa', dilate_frame)
        #     self.frameList.pop(0)
        #     # find the contours around the white segmented areas
        contours, _ = cv2.findContours(
            dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        return []
    

    def draw_bboxes(self, img, boxes, confs, clss):
        """Draw detected bounding boxes on the original image."""
        # print(boxes, clss)
        # f.write(print(len(self.lastContoursList)))
        contours = self.findCountours(img)    
        movBbox = []
        occupied = 0
        # # for i, cnt in enumerate(contours):
        # #     cv2.drawContours(img, contours, i, (0, 0, 255), 3)
        
        for contour in contours:
            # continue through the loop if contour area is less than 500...
            # ... helps in removing noise detection
            if cv2.contourArea(contour) < 500:
                continue
            # get the xmin, ymin, width, and height coordinates from the contours
            (x, y, w, h) = cv2.boundingRect(contour)
            movBbox.append((x, y, x+w, y+h))
            # draw the bounding boxes
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            # print(cl, self.cls_dict, cf, bb, clss)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            color = self.colors[3] if isOverlapExist((x_min, y_min, x_max, y_max), movBbox) else self.colors[1] 
            if color == self.colors[1]:
                occupied += 1
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            txt = '{} {:.2f}'.format(cls_name, cf)
            img = draw_boxed_text(img, txt, txt_loc, self.colors[1])
        return img, occupied
