import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_geo_vo_data(img_path, npz_path):
    if not os.path.exists(img_path) or not os.path.exists(npz_path):
        print("파일 경로를 확인해주세요.")
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    H, W, _ = img.shape
    img = img[H % 32:, :1216]
    if img.shape[0] != 352:
        img = cv2.resize(img, (1216, 352))

    data = np.load(npz_path)
    kpts = data['kpts']    # [N, 2]
    edges = data['edges']  # [2, E]

    plt.figure(figsize=(15, 5))
    plt.imshow(img)
    plt.title(f"Geo-VO Graph Visualization: {os.path.basename(npz_path)}")

    for i in range(edges.shape[1]):
        pt1 = kpts[edges[0, i]]
        pt2 = kpts[edges[1, i]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
                 color='lime', linewidth=0.5, alpha=0.4)

    plt.scatter(kpts[:, 0], kpts[:, 1], 
                c='red', s=2, edgecolors='none', alpha=0.8)

    plt.axis('off')
    plt.tight_layout()
    
    save_name = "graph_check.png"
    plt.savefig(save_name, dpi=200)
    print(f"시각화 완료! '{save_name}' 파일을 확인하세요.")
    plt.show()

if __name__ == "__main__":
    SEQ = "00"
    IMG_IDX = "000100"
    
    RAW_IMG = f"/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset/sequences/{SEQ}/image_2/{IMG_IDX}.png"
    NPZ_DATA = f"/home/jnu-ie/kys/Geo-VO/gendata/precomputed/{SEQ}/image_2/{IMG_IDX}.npz"
    
    visualize_geo_vo_data(RAW_IMG, NPZ_DATA)