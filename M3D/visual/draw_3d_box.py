
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import os
import sys
sys.path.append("/data2/monoDETR/code/monod_best/")
from lib.datasets.kitti.kitti_utils import Object3d, Calibration
from lib.datasets.utils import draw_projected_box3d


def get_objects_from_label(label_file):
    """根据标注数据获取检测对象"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    # print("lines:", lines)
    objects = [Object3d(line) for line in lines if line.split(" ")[0] in "Car"]
    return objects

def img_read(image_path, file_idx):
    """read image
    image_path: str
    file_idx: str, 000001 
    """
    img_file = f"{image_path}/image_3/{file_idx:>06}.png"
    img = cv2.imread(img_file)
    return img

def draw_3D_box(img, file_idx, calib_path, label_path, box_3d=True,
                color=(255, 255, 255), thickness=1, box_bev=False, voxel_size=0.1, ax=None):
    """
    3d boundingbox可视化
    img: cv2对象
    file_index: str, 000001
    calib_path: str
    label_path: str
    box_3d: bool
    color: tuple
    thickness: int
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print("cur_dir", current_dir)
    label_file = f"{label_path}/{file_idx:>06}.txt"
    print(label_file)
    # 如果是测试集，没有标注数据
    if "/testing/" in label_file:
        return img
    objects = get_objects_from_label(label_file)
    
    # print("objects:", len(objects))
    # 如果没有检测到物体直接返回
    if not objects:
        print(f"label_file: {label_file}, No object detected!")
        return img
    
    calib_filename = f"{calib_path}/calib_3/{file_idx:>06}.txt"
    calib = Calibration(calib_filename)
    
    corners_list = []
    
    # 初始化为一个大的正数和小的负数，便于找到最小和最大值
    min_x = float('inf')
    max_x = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')
    
    
    img2 = np.copy(img)  # for 3d bbox
    
    for object in objects:
        
        corners3d = object.generate_corners3d()

        if not corners3d.any():
            print("no corners3d!")
        
        corners_list.append(corners3d)
        # print("corners3d:", corners3d)
        
        if box_bev:
            min_x = min(min_x, np.min(corners3d[:, 0]))
            max_x = max(max_x, np.max(corners3d[:, 0]))
            min_z = min(min_z, np.min(corners3d[:, 2]))
            max_z = max(max_z, np.max(corners3d[:, 2]))
            Object3d.MIN_XZ = np.array([min_x, min_z])
            width = int((max_x - min_x) / voxel_size) + 1
            height = int((max_z - min_z) / voxel_size) + 1
            Object3d.BEV_SHAPE = np.array([height, width])

    
    if box_3d:
        corners3d_N = np.array(corners_list)
        # print("corners3d_N: ", corners3d_N.shape, "\n", corners3d_N)
        boxes, boxes_corner = calib.corners3d_to_img_boxes(corners3d_N)
        # print("boxes:", len(boxes))
        # print("boxes_corner:", len(boxes_corner), boxes_corner)
        for index, box_3d_corner in enumerate(boxes_corner):
            # 3d框角点连线
            img2 = draw_projected_box3d(img2, box_3d_corner, color=color, thickness=1)

            if thickness != 1:
                # 标签文本
                label_text = f"{objects[index].cls_type}: {objects[index].score:.2f}"
                # print("label_text:", label_text)

                # 绘制标签文本
                # print("box_3d_corner:", box_3d_corner)
                x_min = int(np.min(box_3d_corner[:, 0]))
                y_min = int(np.min(box_3d_corner[:, 1]))
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                # print("rectangle:", (x_min, y_min - text_height - baseline), (x_min + text_width, y_min))
                cv2.rectangle(img2, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
                cv2.putText(img2, label_text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

        # # 显示图像
        # plt.switch_backend("agg")
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        # 保存图像
        # plt.imsave(f'{current_dir}/{file_idx:>06}_{color}_box3d.png', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.imsave(f'{current_dir}/{file_idx:>06}_box3d.png', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    

    else:
        # bev视角图
        if 0:
            # 设置画布
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 500)
            ax.set_aspect('equal')
            ax.set_title("BEV Visualization")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            # 绘制每个物体的 2D 边界框
            for obj in objects:
                box2d = obj.to_bev_box2d(oblique=True)
                print("bev_box:", box2d)
                if not box2d.any():
                    print("no bevbox!")
                # 绘制多边形
                edge_color = 'purple' if color==(255, 0, 255) else 'green'
                # print("edge_color:", edge_color)
                polygon = plt.Polygon(box2d, closed=True, fill=None, edgecolor=edge_color)
                ax.add_patch(polygon)
                # 标注中心
                center = ((box2d[:, 0].max() + box2d[:, 0].min()) / 2, (box2d[:, 1].max() + box2d[:, 1].min()) / 2)
                ax.plot(center[0], center[1], 'bo')
            # plt.show()
        else:
            # img2 = np.zeros((80, 80, 3), dtype=np.uint8)  # 假设 600x700 尺寸的 BEV 图像

            # 对每个检测到的物体提取位置和形状信息
            for obj in objects:
                # 方法2 方向角
                x, y, z = obj.pos[0], obj.pos[1], obj.pos[2]
                width, height, length = obj.w, obj.h, obj.l

                # # 计算旋转矩形框的四个角点
                cos_r_y = np.cos(obj.ry)
                sin_r_y = np.sin(obj.ry)
                dx = width / 2
                dz = length / 2
                
                # corners = np.array([
                #     [-dx * cos_r_y + dz * sin_r_y, -dx * sin_r_y - dz * cos_r_y],
                #     [dx * cos_r_y + dz * sin_r_y, dx * sin_r_y - dz * cos_r_y],
                #     [dx * cos_r_y - dz * sin_r_y, dx * sin_r_y + dz * cos_r_y],
                #     [-dx * cos_r_y - dz * sin_r_y, -dx * sin_r_y + dz * cos_r_y]
                # ])

                corners = np.array([
                    [-dx * sin_r_y + dz * cos_r_y, dx * cos_r_y + dz * sin_r_y],
                    [dx * sin_r_y + dz * cos_r_y, -dx * cos_r_y + dz * sin_r_y],
                    [dx * sin_r_y - dz * cos_r_y, -dx * cos_r_y - dz * sin_r_y],
                    [-dx * sin_r_y - dz * cos_r_y, dx * cos_r_y - dz * sin_r_y]
                ])
                # 将角点映射到图像坐标
                img_corners = corners + [x, z]
                
                # 真实值紫色，预测值绿色
                edge_color = 'purple' if color==(255, 0, 255) else 'green'
                # print(f"thickness:{thickness}, color:{color}, edge_color:{edge_color}")

                # 创建旋转矩形
                poly = patches.Polygon(img_corners, closed=True, edgecolor=edge_color, facecolor='none')
                ax.add_patch(poly)
                
                # 将角点映射到图像坐标
                img_corners = corners + [x, z]
                ax.add_patch(poly)

                # 添加标签
                if thickness != 1:
                    label_x, label_z = img_corners[0]  # 使用第一个角点作为标签位置
                    ax.text(label_x, label_z, f"{obj.score:.2f}", color=edge_color, fontsize=4,verticalalignment='bottom', horizontalalignment='left')

                # # 方法1 直角
                # x, y, z = obj.pos[0], obj.pos[2], obj.pos[1]
                # width, height, length = obj.w, obj.h, obj.l
                
                # # Calculate bottom-left corner coordinates for the rectangle
                # bottom_left_x = x - width / 2
                # bottom_left_y = y - length / 2

                # # Create a rectangle patch
                # edge_color = 'purple' if color==(255, 0, 255) else 'green'
                # rect = Rectangle((bottom_left_x, bottom_left_y), width, length, 
                #                 linewidth=1, edgecolor=edge_color, facecolor='none')
                
                # # Add the rectangle to the plot
                # ax.add_patch(rect)

                # if thickness != 1:
                #     # Add label to the rectangle (assuming label is 0.1 as per your request)
                #     ax.text(x, y, f"{obj.score:.2f}", ha='center', va='center', fontsize=4, color=edge_color)

            # Set plot limits and labels
            ax.set_xlim(-50, 50)  # Example limits, adjust according to your data
            ax.set_ylim(0, 60)  # Example limits, adjust according to your data
            ax.set_aspect('equal')  # Ensure aspect ratio is equal
            # ax.grid(True)
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Bird\'s Eye View')
            # 设置轴标签和标题
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('z (meters)')
            ax.set_title('BEV with Rotated Rectangles')

            # plt.show()
            
            # 保存图像
            plt.savefig(f'{current_dir}/{file_idx:>06}_{color}_bev.png', dpi=200)
            # cv2.imwrite(f'{current_dir}/{file_idx:>06}_{color}_bev.png', img2)
    
    return img2

  

def mix_360_bbox():
    output_dir = "/data2/monodetr-0722/train_1/outputs/0730_200/monodetr/outputs/data"
    
    kitti_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset" 
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            file_idx = filename.split(".")[0]
        else:
            continue
        img = img_read(f"{kitti_dir}/training", file_idx)
        print(img) 
        # dt
        img = draw_3D_box(img, file_idx,
                        f"{kitti_dir}/training",
                        output_dir,
                        color=(0, 255, 0), thickness=2)
        # gt 
        img = draw_3D_box(img, file_idx, f"{kitti_dir}/training",
                        f"{kitti_dir}/training/label_3",
                        color=(255, 0, 255), thickness=1)  



def check_ap_bbox():
    output_dir1 = "/data2/monodetr-0722/train_1/outputs/0730_200/monodetr/outputs/data"
    
    output_dir2 = "/data2/monodetr-0722/train_1/outputs/best/monodetr/outputs/data"
    
    kitti_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset" 
    for filename in os.listdir(output_dir1):
        if filename.endswith(".txt"):
            file_idx = filename.split(".")[0]
        else:
            continue
        img = img_read(f"{kitti_dir}/training", file_idx)
        print(img) 
        # dt1
        img = draw_3D_box(img, file_idx,
                        f"{kitti_dir}/training",
                        output_dir1,
                        color=(0, 255, 0), thickness=2)
        # dt2
        img = draw_3D_box(img, file_idx,
                        f"{kitti_dir}/training",
                        output_dir2,
                        color=(255,245, 238), thickness=2)
        # # gt 
        # img = draw_3D_box(img, file_idx, f"{kitti_dir}/training",
        #                 f"{kitti_dir}/training/label_3",
        #                 color=(255, 0, 255), thickness=1)  


if __name__ == "__main__":
    check_ap_bbox()
