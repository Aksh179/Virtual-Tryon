
from flask import Flask, jsonify, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
def poseDetectorshirt(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    width = 368
    height = 368
    inWidth = width
    inHeight = height
    thr = 0.2
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)
        shoulder_left = "LShoulder"
        shoulder_right = "RShoulder"
        hip_left = "LHip"
        hip_right = "RHip"
        x_shoulder_left = x_shoulder_right = x_hip_left = x_hip_right = y_shoulder_left = y_shoulder_right = y_hip_left = y_hip_right = 0



        if shoulder_left in BODY_PARTS and shoulder_right in BODY_PARTS and hip_left in BODY_PARTS and hip_right in BODY_PARTS:
            id_shoulder_left = BODY_PARTS[shoulder_left]
            id_shoulder_right = BODY_PARTS[shoulder_right]
            id_hip_left = BODY_PARTS[hip_left]
            id_hip_right = BODY_PARTS[hip_right]
            
            if not points:
    # Access elements using indices
                x_shoulder_left, y_shoulder_left = points[id_shoulder_left]
                x_shoulder_right, y_shoulder_right = points[id_shoulder_right]
                x_hip_left, y_hip_left = points[id_hip_left]
                x_hip_right, y_hip_right = points[id_hip_right]
            else:
                pass
            x_min = min(x_shoulder_left, x_shoulder_right, x_hip_left, x_hip_right)
            y_min = min(y_shoulder_left, y_shoulder_right, y_hip_left, y_hip_right)
            x_max = max(x_shoulder_left, x_shoulder_right, x_hip_left, x_hip_right)
            y_max = max(y_shoulder_left, y_shoulder_right, y_hip_left, y_hip_right)

            x=x_min
            y=x_max
            w=y_max
            h=y_min
            
            
    return x,y,w,h  
        

def poseDetectorPants(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    width = 368
    height = 368
    inWidth = width
    inHeight = height
    thr = 0.2
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)
        hip_left = "LHip"
        hip_right = "RHip"
        ankle_left = "LAnkle"
        ankle_right = "RAnkle"
        x_hip_left = x_hip_right = x_ankle_left = x_ankle_right = y_hip_left = y_hip_right = y_ankle_left = y_ankle_right = 0

        if hip_left in BODY_PARTS and hip_right in BODY_PARTS and ankle_left in BODY_PARTS and ankle_right in BODY_PARTS:
            id_hip_left = BODY_PARTS[hip_left]
            id_hip_right = BODY_PARTS[hip_right]
            id_ankle_left = BODY_PARTS[ankle_left]
            id_ankle_right = BODY_PARTS[ankle_right]
            
            if not points:
                x_hip_left, y_hip_left = points[id_hip_left]
                x_hip_right, y_hip_right = points[id_hip_right]
                x_ankle_left, y_ankle_left = points[id_ankle_left]
                x_ankle_right, y_ankle_right = points[id_ankle_right]
            else:
                pass
            
            x_min = min(x_hip_left, x_hip_right, x_ankle_left, x_ankle_right)
            y_min = min(y_hip_left, y_hip_right, y_ankle_left, y_ankle_right)
            x_max = max(x_hip_left, x_hip_right, x_ankle_left, x_ankle_right)
            y_max = max(y_hip_left, y_hip_right, y_ankle_left, y_ankle_right)

            # Calculate width and height
            x=x_min
            y=x_max
            w=y_max
            h=y_min
            
    return x, y, w, h


def poseDetectorFrock(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    width = 368
    height = 368
    inWidth = width
    inHeight = height
    thr = 0.2
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)
        shoulder_left = "LShoulder"
        shoulder_right = "RShoulder"
        hip_left = "LHip"
        hip_right = "RHip"
        knee_left = "LKnee"  # Changed ankle to knee
        knee_right = "RKnee"  # Changed ankle to knee
        x_shoulder_left = x_shoulder_right =  y_shoulder_left = y_shoulder_right = x_knee_left = x_knee_right = y_knee_left = y_knee_right  = 0
        

        if shoulder_left in BODY_PARTS and shoulder_right in BODY_PARTS and hip_left in BODY_PARTS and hip_right in BODY_PARTS and knee_left in BODY_PARTS and knee_right in BODY_PARTS:
            id_shoulder_left = BODY_PARTS[shoulder_left]
            id_shoulder_right = BODY_PARTS[shoulder_right]
            id_knee_left = BODY_PARTS[knee_left]  # Added knee ids
            id_knee_right = BODY_PARTS[knee_right]  # Added knee ids
            
            if not points:
                
                
                x_shoulder_left, y_shoulder_left = points[id_shoulder_left]
                x_shoulder_right, y_shoulder_right = points[id_shoulder_right]
                x_knee_left, y_knee_left = points[id_knee_left]  # Added knee points
                x_knee_right, y_knee_right = points[id_knee_right]  # Added knee points
            else:
                pass
            x_min = min(x_shoulder_left, x_shoulder_right, x_knee_left, x_knee_right)
            y_min = min(y_shoulder_left, y_shoulder_right, y_knee_left, y_knee_right)  
            x_max = max(x_shoulder_left, x_shoulder_right,  x_knee_left, x_knee_right)  
            y_max = max(y_shoulder_left, y_shoulder_right,  y_knee_left, y_knee_right)  

            # Calculate width and height
            x=x_min
            y=x_max
            w=y_max
            h=y_min
            
    return x, y, w, h



app = Flask(__name__)

    


@app.route('/predict/',methods=['POST'])
def predict():
    shirtno = int(request.json.get('shirt'))
    pantno = int(request.json.get('pant'))
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=shirtno
    i=pantno
    while True:
        imgarr=["static/shirt1.png",'static/shirt2.png','static/shirt51.jpg','static/shirt6.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #Pixels with intensity values greater than 200 will be set to 255 (white), and pixels with intensity values less than or equal to 200 will be set to 0 (black).#there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        imgarr=["static/pant7.jpg",'static/pant2.png']
        #i=input("Enter the pant number you want to try")
        imgpant = cv2.imread(imgarr[i-1],1)  #1: This argument specifies the flag for reading the image in color mode (i.e., with three color channels: Blue, Green, and Red).
        imgpant=imgpant[:,:,0:3]#This ensures that only the first three color channels (Blue, Green, and Red) are retained, while any additional channels, such as an alpha channel for transparency, are discarded.
        pantgray = cv2.cvtColor(imgpant,cv2.COLOR_BGR2GRAY) #grayscale conversion
        if i==1:
            ret, orig_mask = cv2.threshold(pantgray,100 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_mask_inv = cv2.bitwise_not(orig_mask)
        else:
            ret, orig_mask = cv2.threshold(pantgray,50 , 255, cv2.THRESH_BINARY)
            orig_mask_inv = cv2.bitwise_not(orig_mask)
        origpantHeight, origpantWidth = imgpant.shape[:2]
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
        x1,x2,y1,y2=poseDetectorPants(img)
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
     #   cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)
     #         cv2.namedWindow("img",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        results=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in results:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
            pantWidth =  3 * w  #approx wrt face width
            pantHeight = pantWidth * origpantHeight / origpantWidth #preserving aspect ratio of original image..
            # Center the pant..
            #Taking the coordinate as reference adjust the clothing position 
            if i==1:
                x1 = x-2*w
                x2 =x1+5*w
                y1 = y+4*h
                y2 = y+h*10
            elif i==2:
                x1 = x-2*w
                x2 =x1+5*w
                y1 = y+4*h
                y2 = y+h*10
            else :
                x1 = x-w/2
                x2 =x1+5*w/2
                y1 = y+5*h
                y2 = y+h*14
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            #two cases:
            """
            close to camera: image will be to big
            so face ke x+w ke niche hona chahiye warna dont render at all
            """
            if x1 < 0:
                x1 = 0 #top left ke bahar
            if x2 > img.shape[1]:
                x2 =img.shape[1] #bottom right ke bahar
            if y2 > img.shape[0] :
                y2 =img.shape[0] #nichese bahar
            if y1 > img.shape[0] :
                y1 =img.shape[0] #nichese bahar
            if y1==y2:
                y1=0
            temp=0
            if y1>y2:
                temp=y1
                y1=y2
                y2=temp
            # Re-calculate the width and height of the pant image(to resize the image when it wud be pasted)
            
            
            
            
            pantWidth = int(abs(x2 - x1))
            pantHeight = int(abs(y2 - y1))
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            #cv2.cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            # Re-size the original image and the masks to the pant sizes
            '''
            cv2.resize() is a function in OpenCV used to resize images.
            imgpant is the original pant image.
            (pantWidth, pantHeight) specifies the desired width and height for the resized image.
            interpolation=cv2.INTER_AREA specifies the interpolation method to be used during resizing. In this case, INTER_AREA is used, which is suitable for shrinking or downsampling images. It is typically used when reducing the size of an image.
            '''
            pant = cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
            
        # take ROI for pant from background equal to size of pant image
            roi = img[y1:y2, x1:x2]
                # roi_bg contains the original image only where the pant is not
                # in the region that is the size of the pant.
            num=roi
            roi_bg = cv2.bitwise_and(roi,num,mask = mask_inv)
                # roi_fg contains the image of the pant only where the pant is
            roi_fg = cv2.bitwise_and(pant,pant,mask = mask)
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
                # place the joined image, saved to dst back over the original image
            top=img[0:y,0:resizewidth]
            bottom=img[y+h:resizeheight,0:resizewidth]
            midleft=img[y:y+h,0:x]
            midright=img[y:y+h,x+w:resizewidth]
            blurvalue=5
           # top=cv2.GaussianBlur(top,(blurvalue,blurvalue),0)
            bottom=cv2.GaussianBlur(bottom,(blurvalue,blurvalue),0)
            midright=cv2.GaussianBlur(midright,(blurvalue,blurvalue),0)
            midleft=cv2.GaussianBlur(midleft,(blurvalue,blurvalue),0)
            img[0:y,0:resizewidth]=top
            img[y+h:resizeheight,0:resizewidth]=bottom
            img[y:y+h,0:x]=midleft
            img[y:y+h,x+w:resizewidth]=midright
            img[y1:y2, x1:x2] = dst

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-int(1.5*w)
            x2s =x1s+int(4*w)
            y1s = y+h
            y2s = y1s+int(h*4)
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            
            
            
            
            
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
           
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #Resizes the shirt image (imgshirt) to the specified width and height (shirtWidth and shirtHeight). 
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Shirt Width: {} cm".format(int(shirtWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Shirt Height: {} cm".format(int(shirtHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Pant Width: {} cm".format(int(pantWidth)), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Pant Height: {} cm".format(int(pantHeight)), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            # Your existing code ...
            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
# Endpoint to handle shirt prediction
@app.route('/predict/shirt', methods=['POST'])
def predict_shirt():
    data = request.json
    shirtno = data.get('shirt') 
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=shirtno
    #i=pantno
    while True:
        imgarr=["static/shirt1.png",'static/images/t-shirts/modal-1.png','static/shirt51.jpg','static/shirt6.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
        x1s,x2s,y1s,y2s=poseDetectorshirt(img)
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
        

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-int(1.5*w)
            x2s =x1s+int(4*w)
            y1s = y+h
            y2s = y1s+int(h*4)
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Shirt Width: {} cm".format(int(shirtWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Shirt Height: {} cm".format(int(shirtHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
@app.route('/predict/pant', methods=['POST'])
def predict_pant():
    data = request.json
    pantno = data.get('pant')
    # Here, you would perform your prediction logic based on the pant ID
    # For demonstration purposes, let's just return a dummy response
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    #ih=shirtno
    i=pantno
    while True:
        
        imgarr=["static/pant72.jpg",'static/pant2.png']
        #i=input("Enter the pant number you want to try")
        imgpant = cv2.imread(imgarr[i-1],1)
        imgpant=imgpant[:,:,0:3]#original img in bgr
        pantgray = cv2.cvtColor(imgpant,cv2.COLOR_BGR2GRAY) #grayscale conversion
        if i==1:
            ret, orig_mask = cv2.threshold(pantgray,100 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_mask_inv = cv2.bitwise_not(orig_mask)
        else:
            ret, orig_mask = cv2.threshold(pantgray,50 , 255, cv2.THRESH_BINARY)
            orig_mask_inv = cv2.bitwise_not(orig_mask)
        origpantHeight, origpantWidth = imgpant.shape[:2]
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        x1,x2,y1,y2=poseDetectorPants(img)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
     #   cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)
     #         cv2.namedWindow("img",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
            pantWidth =  3 * w  #approx wrt face width
            pantHeight = pantWidth * origpantHeight / origpantWidth #preserving aspect ratio of original image..

            # Center the pant..just random calculations..
            if i==1:
                x1 = x-2*w
                x2 =x1+5*w
                y1 = y+4*h
                y2 = y+h*10
            elif i==2:
                x1 = x-2*w
                x2 =x1+5*w
                y1 = y+3*h
                y2 = y+h*11
            else :
                x1 = x-w/2
                x2 =x1+5*w/2
                y1 = y+5*h
                y2 = y+h*14
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            #two cases:
            """
            close to camera: image will be to big
            so face ke x+w ke niche hona chahiye warna dont render at all
            """
            if x1 < 0:
                x1 = 0 #top left ke bahar
            if x2 > img.shape[1]:
                x2 =img.shape[1] #bottom right ke bahar
            if y2 > img.shape[0] :
                y2 =img.shape[0] #nichese bahar
            if y1 > img.shape[0] :
                y1 =img.shape[0] #nichese bahar
            if y1==y2:
                y1=0
            temp=0
            if y1>y2:
                temp=y1
                y1=y2
                y2=temp
            """
            if y+h > y1: #agar face ka bottom most coordinate pant ke top ke niche hai
                y1 = 0
                y2 = 0
            """
            # Re-calculate the width and height of the pant image(to resize the image when it wud be pasted)
            pantWidth = int(abs(x2 - x1))
            pantHeight = int(abs(y2 - y1))
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
        
            pant = cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
        # take ROI for pant from background equal to size of pant image
            roi = img[y1:y2, x1:x2]
                # roi_bg contains the original image only where the pant is not
                # in the region that is the size of the pant.
            num=roi
            roi_bg = cv2.bitwise_and(roi,num,mask = mask_inv)
                # roi_fg contains the image of the pant only where the pant is
            roi_fg = cv2.bitwise_and(pant,pant,mask = mask)
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
                # place the joined image, saved to dst back over the original image
            top=img[0:y,0:resizewidth]
            bottom=img[y+h:resizeheight,0:resizewidth]
            midleft=img[y:y+h,0:x]
            midright=img[y:y+h,x+w:resizewidth]
            blurvalue=5
            #top=cv2.GaussianBlur(top,(blurvalue,blurvalue),0)
            bottom=cv2.GaussianBlur(bottom,(blurvalue,blurvalue),0)
            midright=cv2.GaussianBlur(midright,(blurvalue,blurvalue),0)
            midleft=cv2.GaussianBlur(midleft,(blurvalue,blurvalue),0)
            img[0:y,0:resizewidth]=top
            img[y+h:resizeheight,0:resizewidth]=bottom
            img[y:y+h,0:x]=midleft
            img[y:y+h,x+w:resizewidth]=midright
            img[y1:y2, x1:x2] = dst

   
            cv2.putText(img, "Pant Width: {} cm".format(int(pantWidth)), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Pant Height: {} cm".format(int(pantHeight)), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            # Your existing code ...
            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
@app.route('/frock1/', methods=['POST'])
def frock1():
    data = request.json
    shirtno = data.get('frock1')
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=shirtno
    #i=pantno
    while True:
        #Taking 4th frock value
        imgarr=["static/shirt1.png",'static/shirt2.png','static/shirt51.jpg','static/images/Frocks5/3.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
     #   cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)
     #         cv2.namedWindow("img",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
        

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-int(1.5*w)
            x2s =x1s+int(4*w)
            y1s = y+h
            y2s = y1s+int(h*5)
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Frock Width: {} cm".format(int(shirtWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Frock Height: {} cm".format(int(shirtHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            # Your existing code ...
            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
@app.route('/rogue1', methods=['POST'])
def rogue_endpoint1():
    data = request.json
    shirtno = data.get('shirt')
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=shirtno
    #i=pantno
    while True:
        imgarr=["static/shirt1.png",'static/shirt2.png','static/shirt51.jpg','static/shirt6.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
       
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
     #   cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)
     #         cv2.namedWindow("img",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
        

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-w/2
            x2s =x1s+2*w
            y1s = y+h
            y2s = y1s+h*4
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Shirt Width: {} cm".format(int(shirtWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Shirt Height: {} cm".format(int(shirtHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
@app.route('/rogue', methods=['POST'])
def rogue_endpoint():
    data = request.json
    shirtno = data.get('shirt')
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=shirtno
    #i=pantno
    while True:
        imgarr=["static/shirt1.png",'static/shirt2.png','static/shirt51.jpg','static/shirt6.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
       
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        x1s,x2s,y1s,y2s=poseDetectorshirt(img)
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        #img = cv2.cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.cv2.INTER_AREA)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
        

    #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-int(1.5*w)
            x2s =x1s+int(4*w)
            y1s = y+h
            y2s = y1s+int(h*4)
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            # roi_fg contains the image of the shirt only where the shirt is
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            # join the roi_bg and roi_fg
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Shirt Width: {} cm".format(int(shirtWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Shirt Height: {} cm".format(int(shirtHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            # cv2.putText(img, "Pant Width: {}".format(pantWidth), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            # cv2.putText(img, "Pant Height: {}".format(pantHeight), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            # Your existing code ...
            break
        cv2.imshow("img",img)
        #cv2.cv2.setMouseCallback('img',change_dress)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detetced Size'})
@app.route('/frock/', methods=['POST'])
def frock():
    data = request.json
    frockno = data.get('frock')
    cap=cv2.VideoCapture(0)
    cv2.waitKey(1)
    
    ih=frockno
    while True:
        #Taking 4th frock value
        imgarr=["static/shirt1.png",'static/shirt2.png','static/images/t-shirts/modal-6.png','static/images/Frocks5/3.png']

        #ih=input("Enter the shirt number you want to try")
        imgfrock = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            frockgray = cv2.cvtColor(imgfrock,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(frockgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv2.cvtColor(imgfrock,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origfrockHeight, origfrockWidth = imgfrock.shape[:2]
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        height = img.shape[0]
        width = img.shape[1]
        x1s,x2s,y1s,y2s=poseDetectorFrock(img)
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", resizewidth,resizeheight)
    #    cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
            
            

            frockWidth =  3 * w  #approx wrt face width
            frockHeight = frockWidth * origfrockHeight / origfrockWidth #preserving aspect ratio of original image..
            # Center the shirt..just random calculations..
            x1s = x-int(2*w)
            x2s =x1s+int(5*w)
            y1s = y+h
            y2s = y1s+int(h*5)
            # Check for clipping(whetehr x1 is coming out to be negative or not..)

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp
            """
            if y+h >=y1s:
                y1s = 0
                y2s=0
            """
            # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
            frockWidth = int(abs(x2s - x1s))
            frockHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)
            """
            if not y1s == 0 and y2s == 0:
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                masks_inv = cv2.cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            """
            # Re-size the original image and the masks to the shirt sizes
            frock = cv2.resize(imgfrock, (frockWidth,frockHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
            mask = cv2.resize(orig_masks, (frockWidth,frockHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (frockWidth,frockHeight), interpolation = cv2.INTER_AREA)
            # take ROI for shirt from background equal to size of shirt image
            rois = img[y1s:y2s, x1s:x2s]
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            roi_fgs = cv2.bitwise_and(frock,frock,mask = mask)
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            #print "blurring"
            cv2.putText(img, "Frock Width: {} cm".format(int(frockWidth)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, "Frock Height: {} cm".format(int(frockHeight)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            break
        cv2.imshow("img",img)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv2.destroyAllWindows()
    return jsonify({'size_detected': 'Detectedsize'})
@app.route('/')
def indexx():
    return render_template('index.html')
@app.route('/product1')
def product1():
    return render_template('product1.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')
if __name__ == "__main__":
    app.run(debug=True)