import cv2.aruco as aruco

import cv2
import numpy as np
import math
from objloader_simple import *


def main():
    camera_parameters = np.array([
        [1.08355894e+03, 0.00000000e+00, 5.73362291e+02],
        [0.00000000e+00, 1.08500947e+03, 3.45074914e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    # load obj
    obj = OBJ("./dog.obj", swapyz=True)
    cap = cv2.VideoCapture(0)
    # aruco var
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            return 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) >= 1:
            # get projection matrix
            projection = calc_projection(camera_parameters, corners[0])
            # draw aruco marker bound
            frame = aruco.drawDetectedMarkers(frame, corners)
            # render 3d model
            frame = render(frame, obj, projection, False)

        # show result    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

def calc_projection(camera_parameters, corners):
    # find homography
    corners_3d = np.array([[
                [0, 0],
                [200, 0],
                [200, 200],
                [0, 200]
            ]])
    T, mask = cv2.findHomography(corners_3d, corners[0], cv2.RANSAC)
    # calc real external matrix
    T = T * (-1)
    external = np.dot(np.linalg.inv(camera_parameters), T)
    # calc external matrix and transpose
    col_1 = external[:, 0]
    col_2 = external[:, 1]
    col_3 = external[:, 2]

    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.column_stack((rot_1, rot_2, rot_3, translation))

    return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 15
    h = 200
    w = 200

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render in the middle
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        
    return img



if __name__ == '__main__':
    main()
