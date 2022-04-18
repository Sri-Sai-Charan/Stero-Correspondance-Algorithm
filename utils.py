import numpy as np
import cv2 as cv


def normalize_val(uv):

    uv_d = np.mean(uv, axis=0)
    u_d ,v_d = uv_d[0], uv_d[1]

    u_c = uv[:,0] - u_d
    v_c = uv[:,1] - v_d

    s = (2/np.mean(u_c**2 + v_c**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_d],[0,1,-v_d],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def get_f_matrix(feature_matches):
    normalised = True

    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize_val(x1)
            x2_norm, T2 = normalize_val(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def errorF(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1_temp=np.array([x1[0], x1[1], 1]).T
    x2_temp=np.array([x2[0], x2[1], 1])

    er = np.dot(x1_temp, np.dot(F, x2_temp))
    
    return np.abs(er)


def get_ransac_Inliers(features):
    n_iterations = 1000
    error_thresh = 0.02
    inliers_thresh = 0
    selected_indices = []
    chosen_f = 0

    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = get_f_matrix(features_8)
        for j in range(n_rows):
            feature = features[j]
            error = errorF(feature, f_8)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            selected_indices = indices
            chosen_f = f_8

    filtered_features = features[selected_indices, :]
    return chosen_f, filtered_features

def get_e_matrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corr = np.dot(U,np.dot(np.diag(s),V))
    return E_corr


















def reproject3DPoints(R, C, K, pts3D_4):

    k = 3
    I = np.identity(3)
    P = np.dot(K, np.dot(R[k], np.hstack((I, -C[k].reshape(3,1)))))

    X = pts3D_4[k]
    x_ = np.dot(P, X)
    x_ = x_/x_[2,:]

    x = x_[0, :].T
    y = x_[1, :].T
    return x, y


def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def get3DPoints(K1, K2, matched_pairs, R2, C2):
    pts3D_4 = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        pts3D = []
        x1 = matched_pairs[:,0:2].T
        x2 = matched_pairs[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv.triangulatePoints(P1, P2, x1, x2)  
        pts3D_4.append(X)
    return pts3D_4

def getEpipolarLines(set1, set2, F, image0, image1, file_name, rectified = False):
    # set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = getX(line2, y2_min)
            x2_max = getX(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = getX(line1, y1_min)
            x1_max = getX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]



        cv.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    image_1, image_2 = makeImageSizeSame([img_epi1, img_epi2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv.resize(concat, (1920, 660))
    displaySaveImage(concat, file_name)
    # cv.imshow("a", concat)
    # cv.imwrite("epilines.png", concat)
    # cv.waitKey()
    # cv.destroyAllWindows()
    return lines1, lines2











def get_dataset(dataset_path, n_img):
    print("Reading images from ", dataset_path)
    images = []
    for n in range(0, n_img):
        img_path = dataset_path + "/" + "im" + str(n) + ".png"
        image = cv.imread(img_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def makeImageSizeSame(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized

def showMatches(img_1, img_2, matched_pairs, color, file_name):

    image_1 = img_1.copy()
    image_2 = img_2.copy()

    image_1, image_2 = makeImageSizeSame([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if matched_pairs is not None:
        corners_1_x = matched_pairs[:,0].copy().astype(int)
        corners_1_y = matched_pairs[:,1].copy().astype(int)
        corners_2_x = matched_pairs[:,2].copy().astype(int)
        corners_2_y = matched_pairs[:,3].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 2)
    
    if file_name is not None:      
        cv.imshow(file_name, concat)
        cv.waitKey() 
        cv.destroyAllWindows()
        cv.imwrite(file_name, concat)
    


















def displaySaveImage(image, file_name =  None):
    cv.imshow("image", image)
    cv.waitKey()
    if file_name is not None:
        cv.imwrite(file_name, image)
    cv.destroyAllWindows()



def SSD(mat1, mat2):
    diff_sq = np.square(mat1 - mat2)
    ssd = np.sum(diff_sq)
    return ssd

def SAD(mat1, mat2):
    return np.sum(abs(mat1 - mat1))

def NCC(patch1, patch2):
  patch1_hat, patch2_hat = patch1 - patch1.mean(), patch2 - patch2.mean()
  upper = np.sum(patch1_hat*patch2_hat)
  lower = np.sqrt(np.sum(patch1_hat*patch1_hat)*np.sum(patch2_hat*patch2_hat))
  return upper/lower

















def siftFeatures2Array(sift_matches, kp1, kp2):
    matched_pairs = []
    for i, m1 in enumerate(sift_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    return matched_pairs


    

def getPositiveZCount(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positiveZ = 0
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            n_positiveZ += 1

    return n_positiveZ

# def getX(line, y):
#     x = -(line[1]*y + line[2])/line[0]
#     return x
    
def get_camera_pose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C