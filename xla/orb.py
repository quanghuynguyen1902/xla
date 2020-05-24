import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('./box.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('./box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# display the keypoints and features detected on both images
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv.drawKeypoints(img1,kp1,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv.drawKeypoints(img2,kp2,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)

plt.savefig("./features_orb"+'.jpeg', bbox_inches='tight', 
            dpi=300, optimize=True, format='jpeg')
plt.show()

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.axis('off')
plt.savefig("./orb"  +'.jpeg', bbox_inches='tight', dpi=300, optimize=True, format='jpeg')
plt.show()