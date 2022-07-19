# import cv2
# # img=cv2.imread('img/test.png')

# video = cv2.VideoCapture('video/test.avi')
# success,frame = video.read()
# img = frame

# list_point = []

# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         print(x,y)
#         list_point.append([x,y])
#         cv2.circle(img, (x, y), 2, (0, 0, 255))
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
#         cv2.imshow("image", img)
        
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# while(1):
#     cv2.imshow("image", img)
#     key = cv2.waitKey(5) & 0xFF
#     if key == ord('q'):
#         break
# print(list_point)
# cv2.destroyAllWindows()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device) #cpu


# import torch
# flag = torch.cuda.is_available()
# print(flag)

# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print(device)
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3,3).cuda())


# import torch
# print(torch.__version__)