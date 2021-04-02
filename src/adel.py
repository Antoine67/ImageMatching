import cv2
import numpy as np

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = ((w-1) // 2.0, (h-1)// 2.0)


    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    print (nW, nH)
    
    # adjust the rotation matrix to take into account translation
    M[0, 2] += ((nW-1) / 2.0) - cX
    M[1, 2] += ((nH-1) / 2.0) - cY
    
    # perform the actual rotation and return the image
    return M, cv2.warpAffine(image, M, (nW, nH))

#function that calculates the updated locations of the coordinates
#after rotation
def rotated_coord(points,M):
    points = np.array(points)
    ones = np.ones(shape=(len(points),1))
    points_ones = np.concatenate((points,ones), axis=1)
    transformed_pts = M.dot(points_ones.T).T
    return transformed_pts

#READ IMAGE & CALL FCT
img = cv2.imread("../storage/dataset/dataset1/img_1.png")
points = np.array([[1000,  384]])
#rotate by 90 angle for example
M, rotated = rotate_bound(img, 45)
#find out the new locations
transformed_pts = rotated_coord(points,M)
cv2.imwrite("adel.png", rotated)
print(transformed_pts)