
import numpy as np
import cv2 as cv
from utils import *
import matplotlib
matplotlib.use('Agg')
import argparse
from matplotlib import pyplot as plt

from alive_progress import alive_bar

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Dataset', default='curule', help='data name of the dataset being used, Default: curule')

    Args = Parser.parse_args()
    dataset_name = Args.Dataset
    save_name = dataset_name
    if dataset_name == "curule":
  
        dataset_name = "./media/"+dataset_name
        K_1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        baseline=177.288
        f = K_1[0,0]

    elif dataset_name == "octagon":
    
        dataset_name = "./media/"+dataset_name
        K_1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        K2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        baseline=221.76
        f = K_1[0,0]

    elif dataset_name == "pendulum":

        dataset_name = "./media/"+dataset_name
        K_1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        baseline=537.75
        f = K_1[0,0]

    else:
        print("invalid dataset name entered")
        exit()

    images = get_dataset(dataset_name, 2)
    sift = cv.xfeatures2d.SIFT_create()
    img_0 = images[0].copy()
    img_1 = images[1].copy()

    img_0_gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY) 
    img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img_0_gray, None)
    kp2, des2 = sift.detectAndCompute(img_1_gray, None)

    bf = cv.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]
    print("Matches found!")
    matched_pairs = siftFeatures2Array(chosen_matches, kp1, kp2)
    print("Estimating F & E matrices")
    F_best, matched_pairs_inliers = get_ransac_Inliers(matched_pairs)

    E = get_e_matrix(K_1, K2, F_best)
    R2, C2 = get_camera_pose(E)
    pts3D_4 = get3DPoints(K_1, K2, matched_pairs_inliers, R2, C2)

    z_count1 = []
    z_count2 = []

    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    for i in range(len(pts3D_4)):
        pts3D = pts3D_4[i]
        pts3D = pts3D/pts3D[3, :]
        x = pts3D[0,:]
        y = pts3D[1, :]
        z = pts3D[2, :]    

        z_count2.append(getPositiveZCount(pts3D, R2[i], C2[i]))
        z_count1.append(getPositiveZCount(pts3D, R1, C1))

    z_count1 = np.array(z_count1)
    z_count2 = np.array(z_count2)

    count_thresh = int(pts3D_4[0].shape[1] / 2)
    idx = np.intersect1d(np.where(z_count1 > count_thresh), np.where(z_count2 > count_thresh))
    R2_Matrix = R2[idx[0]]
    C2_Matrix = C2[idx[0]]
    X_ = pts3D_4[idx[0]]
    X_ = X_/X_[3,:]
    print("Estimated R & C as \n", R2_Matrix, "\n", C2_Matrix)


    set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = getEpipolarLines(set1, set2, F_best, img_0, img_1, "Output/epilines_" + save_name+ ".png", False)

    h1, w1 = img_0.shape[:2]
    h2, w2 = img_1.shape[:2]
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), F_best, imgSize=(w1, h1))
    print("Estimated H1 & H2 as", H1, H2)

    img1_rect = cv.warpPerspective(img_0, H1, (w1, h1))
    img2_rect = cv.warpPerspective(img_1, H2, (w2, h2))

    set1_rectified = cv.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
    set2_rectified = cv.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

    img1_rect_draw = img1_rect.copy()
    img2_rect_draw = img2_rect.copy()

    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))

    lines1_rectified, lines2_recrified = getEpipolarLines(set1_rectified, set2_rectified, F_rectified, img1_rect, img2_rect, "Output/rectified_epilines_" + save_name+ ".png", True)

    img1_rect_reshaped = cv.resize(img1_rect, (int(img1_rect.shape[1] / 4), int(img1_rect.shape[0] / 4)))
    img2_rect_reshaped = cv.resize(img2_rect, (int(img2_rect.shape[1] / 4), int(img2_rect.shape[0] / 4)))

    img1_rect_reshaped = cv.cvtColor(img1_rect_reshaped, cv.COLOR_BGR2GRAY)
    img2_rect_reshaped = cv.cvtColor(img2_rect_reshaped, cv.COLOR_BGR2GRAY)

    window = 11

    left_array, right_array = img1_rect_reshaped, img2_rect_reshaped
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    
    h, w = left_array.shape
    disparity_map = np.zeros((h, w))

    x_new = w - (2 * window)

    with alive_bar(h-(window*2),bar ='filling',title = 'Progress :',length = 60) as bar:
        for y in range(window,h-window):
            bar()
            window_left_array = []
            window_right_array = []
            for x in range(window, w-window):
                window_left = left_array[y:y + window, x:x + window]
                window_left_array.append(window_left.flatten())
                window_right = right_array[y:y + window, x:x + window]
                window_right_array.append(window_right.flatten())
            window_left_array = np.array(window_left_array)
            window_left_array = np.repeat(window_left_array[:,:, np.newaxis], x_new, axis=2)
            window_right_array = np.array(window_right_array)
            window_right_array = np.repeat(window_right_array[:,:, np.newaxis], x_new, axis=2)
            window_right_array = window_right_array.T
            abs_diff = np.abs(window_left_array - window_right_array)
            sum_abs_diff = np.sum(abs_diff, axis = 1)
            idx = np.argmin(sum_abs_diff, axis = 0)
            disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
            disparity_map[y, 0:x_new] = disparity 




    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('Output/disparity_image_heat_' +save_name+ ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('Output/disparity_image_gray_' +save_name+ ".png")

    depth = (baseline * f) / (disparity_map + 1e-10)
    depth[depth > 100000] = 100000

    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('Output/depth_image_heat_' +save_name+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('Output/depth_image_gray_' +save_name+ ".png")

if __name__ == "__main__":
    main()




