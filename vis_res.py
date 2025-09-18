import os
import numpy as np
import laspy
from open3d.ml.vis import Visualizer, LabelLUT
import re
import matplotlib.pyplot as plt

# ---------------------- 1. 定义 Label LUT ----------------------
MY_LABELS = {
    1: "Unclassified",
    6: "Building",
    12: "Road",
    16: "Wire",
    17: "Tower",
}

LOG_DIR = "/home/zxhc/Workspace/Open3D-ML/scripts/logs/RandLANet_Custom3D_torch/"
SAVE_FIG_DIR = "/home/zxhc/Workspace/Open3D-ML/"


def create_lut(labels_dict):
    lut = LabelLUT()
    for k, v in labels_dict.items():
        lut.add_label(v, k)
    return lut


# ---------------------- 2. 可视化函数 ----------------------
def visualize_pointcloud(points, labels, lut, name="cloud"):
    v = Visualizer()
    v.set_lut("pred", lut)
    pc_dict = {"points": points, "pred": labels, "name": name}
    v.visualize([pc_dict])


# ---------------------- 3. 保存为 TXT ----------------------
def save_txt(points, labels, save_path):
    save_array = np.hstack([points[:, :3], labels.reshape(-1, 1)])
    np.savetxt(save_path, save_array, fmt='%f %f %f %d')
    print(f"TXT文件保存成功: {save_path}")


# ---------------------- 4. 保存为 LAS ----------------------
def save_las(points, labels, save_path):
    labels = labels.astype(np.uint8)
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = labels

    # n_points = points.shape[0]
    # las.red = np.full(n_points, 100, dtype=np.uint16)
    # las.green = np.full(n_points, 100, dtype=np.uint16)
    # las.blue = np.full(n_points, 100, dtype=np.uint16)

    las.write(save_path)
    print(f"LAS文件保存成功: {save_path}")


# ---------------------- 5. 处理目录中的点云 ----------------------
def process_pointclouds(data_dir,
                        label_dir,
                        save_dir_txt,
                        save_dir_las,
                        visualize=False,
                        save_txt_flag=False,
                        save_las_flag=True):
    os.makedirs(save_dir_txt, exist_ok=True)
    os.makedirs(save_dir_las, exist_ok=True)

    lut = create_lut(MY_LABELS)

    for fname in os.listdir(data_dir):
        if not fname.endswith(".npy"):
            continue

        data_path = os.path.join(data_dir, fname)
        label_path = os.path.join(label_dir, fname)

        if not os.path.exists(label_path):
            print(f"Label 文件不存在，跳过: {fname}")
            continue

        # 读取数据
        points = np.load(data_path)
        pred = np.load(label_path, allow_pickle=True)

        # 转换标签：字符串转索引，如果是整数则保持
        pred_labels = np.array([
            list(MY_LABELS.keys())[list(MY_LABELS.values()).index(l)]
            if isinstance(l, str) else l for l in pred
        ])
        # pred_labels = np.zeros(points.shape[0], dtype=np.int32)

        base_name = os.path.splitext(fname)[0]

        if save_txt_flag:
            save_txt(points, pred_labels,
                     os.path.join(save_dir_txt, base_name + ".txt"))

        if save_las_flag:
            save_las(points, pred_labels,
                     os.path.join(save_dir_las, base_name + ".las"))

        if visualize:
            visualize_pointcloud(points, pred_labels, lut, name=base_name)


# ---------------------- 6. 主函数 ----------------------
def main():
    data_dir = "/home/zxhc/Workspace/Open3D-ML/data/Custom3D/test/"
    label_dir = "/home/zxhc/Workspace/Open3D-ML/res/labels"
    save_dir_txt = "/home/zxhc/Workspace/Open3D-ML/res/txt/"
    save_dir_las = "/home/zxhc/Workspace/Open3D-ML/res/las/"

    process_pointclouds(data_dir,
                        label_dir,
                        save_dir_txt,
                        save_dir_las,
                        visualize=False,
                        save_txt_flag=False,
                        save_las_flag=True)


if __name__ == "__main__":
    main()
