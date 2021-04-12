import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_deskewed_word(input_word_img):    #Returns the deskewed word as a numpy array
    word_img = input_word_img.copy()

    word_img = 255-word_img
    
    minLineLength = word_img.shape[1]
    maxLineGap = 20
    #longest_line_index=0
    longest_line=[0,0]
    confiedence = 100
    lines = None

    while(lines is None):
        lines = cv2.HoughLinesP(word_img,2,np.pi/180,confiedence,minLineLength,maxLineGap)
        #print len(lines)
        confiedence -= 5
        if(confiedence == 50):
            break
    if(not(lines is None)):
        for loc,line in enumerate(lines):
            for x1,y1,x2,y2 in line:
                dist = np.sqrt(abs((x2-x1)^2-(y2-y1)^2))
                angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi
                if(dist > longest_line[0] and angle < 30 and angle > -30):
                    longest_line[0] = dist
                    longest_line[1] = angle
                    

        rows,cols = word_img.shape
        rot = cv2.getRotationMatrix2D((cols/2,rows/2),longest_line[1],1)
        rotated = 255 - cv2.warpAffine(word_img,rot,(cols,rows),cv2.INTER_CUBIC)

        rotated[rotated>=64]=255
        rotated[rotated<64]=0
        return rotated
    else:
        return None



def get_rect_rank(rect):
    x_mean=(rect[0]+rect[2])/2
    y_mean=(rect[1]+rect[3])/2
    rank = (y_mean/150)*100000+x_mean
    return rank

def get_rect_area(rect):
    return abs(rect[0]-rect[2])*abs(rect[1]-rect[3])


def merge_nearby_rectangles(list_rect_coordinates):
    size =len(list_rect_coordinates)
    res=[]
    is_overlap=np.array(list(range(0,size)))
    for i in range(0,size):
        for j in range(i,size):
            area_1 = get_rect_area(list_rect_coordinates[i])
            area_2 = get_rect_area(list_rect_coordinates[j])
            area_ratio = float(area_1)/area_2 if area_1 > area_2 else float(area_2)/area_1 
            if(area_ratio > 4  and overlap(list_rect_coordinates[i],list_rect_coordinates[j],2)):
                is_overlap[j]=is_overlap[i]
                is_overlap[i]=is_overlap[j]
    flag = False
    for i in range(len(is_overlap)):
        rect = list_rect_coordinates[i]
        for j in np.where(is_overlap==i)[0]:
            rect=union(rect,list_rect_coordinates[j])
            flag=True
        if(flag):
            res.append(rect)
            flag=False
    return res
def union(r1,r2):
    r1_left   = r1[0]
    r1_right  = r1[2]
    r1_bottom = r1[3]
    r1_top    = r1[1]
    
    r2_left   = r2[0]
    r2_right  = r2[2]
    r2_bottom = r2[3]
    r2_top    = r2[1]


    x = min(r1_left,r2_left)
    y = min(r1_top,r2_top)
    x_w = max(r1_right,r2_right)
    y_h = max(r1_bottom,r2_bottom)

    return [x,y,x_w,y_h]
def overlap(r1,r2,bias):

    r1_left   = r1[0]
    r1_right  = r1[2]
    r1_bottom = r1[3] + bias
    r1_top    = r1[1] - bias
    
    r2_left   = r2[0]
    r2_right  = r2[2]
    r2_bottom = r2[3] + bias
    r2_top    = r2[1] - bias

    h_overlaps = (r1_left <= r2_right) and (r1_right >= r2_left)
    v_overlaps = (r1_bottom >= r2_top) and (r1_top <= r2_bottom)
    return h_overlaps and v_overlaps


def get_word_coordinates(input_img,debug=False): #Returns list of coordinates as a list. It contains [x,y,x+w,y+h]
    binary_img=input_img.copy()         #Original image is not modified

    binary_img[binary_img>=128]=255
    binary_img[binary_img<128]=0

    binary_img_area=binary_img.shape[0]*binary_img.shape[1]

    out_binary_img =binary_img.copy()
    debug_binary_img= binary_img.copy()

    binary_img = 255 - binary_img

    contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    word_coord = []
    
    k=0
    cnt=0

    c_max = max(contours, key = cv2.contourArea)
    for (i, j) in zip(contours, hierarchy[0]):
        if cv2.contourArea(i)==cv2.contourArea(c_max):
            x2,y2,w2,h2 = cv2.boundingRect(i)
            word_coord.append([x2,y2,x2+w2,y2+h2])
    word_coord.sort(key=lambda x:get_rect_rank(x))
    word_coord=merge_nearby_rectangles(word_coord)
    if(debug):
        for x,y,x_w,y_h in word_coord:
                debug_binary_img = cv2.rectangle(debug_binary_img,(x,y),(x_w,y_h),(0,0,0),2)
                cv2.putText(debug_binary_img,str(cnt),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cnt += 1
        cv2.imwrite("contours.jpg",debug_binary_img)
    return word_coord

def get_word_image(img,rect):
    x,y,x_w,y_h=rect
    return img[y-5:y_h,x-5:x_w+5]



def shadow_removal(img):
  rgb_planes = cv2.split(img)

  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 21)
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)
  result_norm = cv2.merge(result_norm_planes)

  return result_norm



def final_processing(img):
    img = shadow_removal(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.medianBlur(gray, 13)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    deskw = get_deskewed_word(thresh)
    
    deskwcpy = deskw.copy()
    kernel = np.ones((10,10), dtype=np.uint8)
    deskw= cv2.erode(deskw, kernel,iterations=1)
    words=get_word_coordinates(deskw,False)
    cropped = deskwcpy[words[0][1]:words[0][3], words[0][0]:words[0][2]]
    cropped = cv2.copyMakeBorder(cropped, 10, 10, 30, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    mid = cropped.shape[1]/2
    orig_crop = cropped.copy()
    for i in range(cropped.shape[0]):
      px = cropped[i, int(mid)]
      if(px==0):
        blpx = i
        break

    cv2.rectangle(cropped, (0, blpx-20), (cropped.shape[1], blpx+18), (255, 255, 0), -1)
    

    kernel = np.ones((3,3), dtype=np.uint8)
    kernel2 = np.ones((3,3), dtype=np.uint8)
    cropped = cv2.erode(cropped, kernel,iterations=1)
    orig_crop = cv2.erode(orig_crop, kernel2,iterations=1)
    contours, hierarchy = cv2.findContours(cropped,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    characters = []

    for (i, j) in zip(contours, hierarchy[0]):
      if(cv2.contourArea(i)>1000 and j[3]!=-1):
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(cropped, (x,y), (x+w, y+h), (0,255,0), 2)
        characters.append(orig_crop[0:orig_crop.shape[1],x:x+w])
    characters.reverse()

    return characters
