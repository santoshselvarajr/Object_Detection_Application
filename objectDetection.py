#Loading required libraries
import os #Change the directory
#Set the working directory
os.chdir("C:\\Users\\Santosh Selvaraj\\Documents\\Working Directory\\Computer_Vision_A_Z_Template_Folder\\Module 2 - Object Detection")
import torch #Efficient library to build neural networks | Dynamic graph structure
from torch.autograd import Variable #Convert tensors to torch variables
import cv2 #Draw rectangles around the objects
#Convert input image to be compatible with neural network
#VOC Classes gives the number of the classes
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd #Constructor for SSD neural network
import imageio #Process images of the video
import time
#from gtts import gTTS
#from pygame import mixer

#Track time when it begins
processStart = time.time()

#Write the Alert audio
#alert = "Alert"
#response = gTTS(text=alert,lang="en",slow=False)
#response.save("alert.mp3")
#mixer.init()
#mixer.music.load("alert.mp3")

#Defining functions to detect the objects
def detect(frame,net,transform):
    height, width = frame.shape[:2] #Get the height and width from the shape attribute
    frame_t = transform(frame)[0] #Transform frame to desired input structure which is given by the 1st element
    x = torch.from_numpy(frame_t).permute(2,0,1) #Convert to torch format from numpy format and permute to a different RGB
    x = Variable(x.unsqueeze(0)) #Add the batch dimension and convert to a torch variable which holds the weights additionally
    y = net(x) #Feed the input to the neural network
    #Detections = [batch,class type,number of occurrence,(score,x0,y0,x1,y1)]
    detections = y.data #Get value of outputs
    scale = torch.Tensor([width,height,width,height])
    #Iterate and create boxes
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0,i,j,1:] * scale).numpy()
            distance = int(pt[2])-int(pt[0]) + int(pt[3])-int(pt[1])
            cv2.putText(frame,labelmap[i-1].title(),(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame,str(round(100/distance,2)),(int(pt[2])-3,int(pt[3])+3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            if distance < 400:
                cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(0,255,0), 2)
            elif distance < 500:
                cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,255,0), 2)
            else:
                cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0), 3)
                loc1 = int(pt[0] + (pt[2] - pt[0])/4)
                loc2 = int((pt[1] + pt[3])/2)
                cv2.putText(frame,"ALERT!!!",(loc1,loc2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            j += 1
    return frame

#Create the SSD Neural Network
net = build_ssd('test') #Initialize the neural network
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage)) #Load the weights from the pre-built model

#Creating the transformation
transform = BaseTransform(net.size,(104/256.0,117/256.0,123/256.0)) #Scale transform to get the original video

#Doing object detection on a video
reader = imageio.get_reader("Test-2_Trim.mp4") #Read the video
fps = reader.get_meta_data()['fps'] #Get the frames per second
writer = imageio.get_writer('output.mp4',fps = fps) #Create a new video with the same framespersecond
for i, frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()

#Track time when it ends
processEnd = time.time()

#Total time elapsed
print("Time Taken for this endeavour: %s minutes..." % (round((processEnd - processStart)/60,2)))