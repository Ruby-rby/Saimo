import os
import random
import re
import shutil

def rename_and_move_images(src_dir, dst_dir, rename_dict):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 遍历所有子目录和文件
    for root, _, files in os.walk(src_dir):
        # /data2/monodetr-0722/kitti360_0719/KITTI-360/data_2d_raw/2013_05_28_drive_0003_sync/image_00/data_rect/0000000000.png
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 可以根据需要添加更多的图片扩展名
                src_path = os.path.join(root, file)
                # pre_dir_name = "2013_05_28_drive_0003_sync"
                # 正则表达式模式匹配出 pre_dir_name
                pattern = r'(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)'
                # 使用re.search来匹配模式
                match = re.search(pattern, src_path)
                # 如果匹配成功，提取匹配的字符串
                if match:
                    pre_dir_name = match.group(1)
                    print("匹配的子字符串:", pre_dir_name)
                else:
                    print("没有匹配到子字符串") 
                key = f"{pre_dir_name}-{os.path.splitext(file)[0]}"  # 获取文件名（不包括扩展名）
                
                if key in rename_dict:
                    new_name = rename_dict[key] + '.png'
                    dst_path = os.path.join(dst_dir, new_name)
                    shutil.copy(src_path, dst_path)  # 复制文件到目标目录
                    print(f"Copied and renamed {src_path} to {dst_path}")
                else:
                    print(f"Key '{key}' not found in rename_dict")
    
def read_lines(path, strip= True):
    with open(path) as f:
        lines = f.readlines()

    if strip:
        # you may also want to remove whitespace characters like `\n` at the end of each line
        lines = [x.rstrip('\n') for x in lines]

    return lines
        
def get_frame_to_kitti_index_dict():
    """{"2013_05_28_drive_0004_sync-0000009808": "000000", "2013_05_28_drive_0006_sync-0000008584":"068889"}"""
    inp_id_list_path = "/data2/bk-monodetr-0722/monodetr-0722/kitti360_0719/SeaBird-main/PanopticBEV/data/kitti_360/ImageSets/org_trainval_det_clean.txt"
    out_id_list_path = "/data2/bk-monodetr-0722/monodetr-0722/kitti360_0719/SeaBird-main/PanopticBEV/data/kitti_360/ImageSets/trainval_det.txt"
    inp_id_list = read_lines(inp_id_list_path)
    out_id_list = read_lines(out_id_list_path)
    
    frame2idx = {}
    for index, val in enumerate(inp_id_list):
        key = val.replace(";", "-")
        frame2idx[key] = out_id_list[index]
    return frame2idx

def copy_files(src_dir, dst_dir):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 遍历源目录下的所有文件
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

def replace_third_line(file_path, new_lines):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) < 1:
        print(f"File {file_path} does not have enough lines.")
        return

    new_lines[2] = lines[0].strip("\n")
    
    print(f"old:{lines}\n new:{new_lines}")

    with open(file_path, 'w') as file:
        file.writelines([line + "\n" for line in new_lines])

def process_files(src_dir, dst_dir, new_lines):
    copy_files(src_dir, dst_dir)
    
    # 遍历源目录下的所有文件
    for file_name in os.listdir(dst_dir):
        dst_file = os.path.join(dst_dir, file_name)
        
        if os.path.isfile(dst_file):
            replace_third_line(dst_file, new_lines)
            print(f"Processed {dst_file}")


def get_max_suffix_number(directory):
    max_suffix = -1
    for file_name in os.listdir(directory):
        # 文件名格式为 "007402.txt"
        try:
            suffix = int(os.path.splitext(file_name)[0])
            if suffix > max_suffix:
                max_suffix = suffix
        except ValueError:
            continue
    return max_suffix

def copy_and_rename_files(src_dir, dst_dir, start_suffix):
    # current_suffix = start_suffix + 1
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        
        dst_file_name = f"{int(os.path.splitext(file_name)[0])+start_suffix+1:06d}{os.path.splitext(file_name)[1]}"
        dst_file = os.path.join(dst_dir, dst_file_name)
        
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
            # current_suffix += 1

def get_kitti_idx(file_path):
    # 读取文件并处理数据
    # 创建一个空的集合来保存处理后的数据
    result_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行尾的换行符
            if not line:
                continue
            original_value = int(line.strip())
            # 加上68890
            new_value = original_value + 68890
            # 将结果添加到集合中
            result_set.add(new_value)
    # 输出集合中的元素
    print(result_set)
    return result_set
    

def generate_imageset():
    # 文件目录路径
    image_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/image_2"
    train_file = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/ImageSets/train.txt"
    val_file = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/ImageSets/val.txt"

    # 获取所有文件名，并转换为整数
    file_names = [int(f.split('.')[0]) for f in os.listdir(image_dir) if f.endswith('.png')]
    file_names.sort()
    
    # 指定范围和数量
    range1 = [0, 68890]
    range2 = [68890, 76367]
    
    # 获取kitti的split结果
    k_train_file_path = '/data2/img_data/data_kitti/KITTIDataset/ImageSets/train.txt'
    k_train_set = get_kitti_idx(k_train_file_path)
    
    
    k_val_file_path = '/data2/img_data/data_kitti/KITTIDataset/ImageSets/val.txt'
    k_val_set = get_kitti_idx(k_val_file_path)
    
    # 训练的train.txt去除
    test_set = set()
    seq_frame2idx = get_frame_to_kitti_index_dict()
    for key, value in seq_frame2idx.items():
        seq = key.split("-")[0]
        if seq == "2013_05_28_drive_0010_sync":
            # print(f"seq:{seq}, idx:{value}")
            test_set.add(int(value))
            continue
    print(f"len 2013_05_28_drive_0010_sync {len(test_set)}")
    # 在指定范围内筛选文件名
    files_in_range1 = [f for f in file_names if range1[0] <= f <= range1[1] and f not in test_set]
    print(len(files_in_range1), files_in_range1[:100])
    # files_in_range2 = [f for f in file_names if range2[0] <= f <= range2[1]]

    # # 随机选择50%的文件
    # train_files1 = random.sample(files_in_range1, int(len(files_in_range1) * 0.5))
    # train_files2 = random.sample(files_in_range2, int(len(files_in_range2) * 0.5))
    train_files1 = files_in_range1[:3000]
    train_files2 = list(k_train_set)

    # 其余的文件
    # val_files1 = list(set(files_in_range1) - set(train_files1))
    # val_files2 = list(set(files_in_range2) - set(train_files2))
    # val_files1 = files_in_range1[8000:16000]
    val_files1 = list()
    val_files2 = list(k_val_set)
    
    # 合并结果
    train_files = train_files1 + train_files2
    val_files = val_files1 + val_files2

    # 将文件名转换回六位字符串并写入文件
    with open(train_file, 'w') as f:
        for file in sorted(train_files):
            f.write(f"{file:06d}\n")

    with open(val_file, 'w') as f:
        for file in sorted(val_files):
            f.write(f"{file:06d}\n")



if __name__ == "__main__":
    # # copy img
    # print("start to copy img")
    # src_dir = "/data2/kitti360/KITTI-360/data_2d_raw/"
    # dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/image_3"
    # rename_dict = get_frame_to_kitti_index_dict()
    # rename_and_move_images(src_dir, dst_dir, rename_dict)
    
    # # copy label
    # print("start to copy label")
    # src_dir = "/data2/monodetr-0722/train_1/data/data_360/training/label"
    # dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/label_3"
    # copy_files(src_dir, dst_dir)
    
#     # copy calib
#     print("start to copy calib")
#     src_dir = "/data2/monodetr-0722/train_1/data/data_360/training/calib"
#     dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/calib_3"
    
#     copy_files(src_dir, dst_dir)
#     new_lines = """P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
# P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
# R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
# Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
# Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01""".split("\n")
    
#     print("start to copy process_files")
#     process_files(src_dir, dst_dir, new_lines)
    
#     # rename kitti files
    
#     # 获取label目录下后缀数字最大的文件名
#     max_suffix = get_max_suffix_number(dst_dir)
    
#     # 将kitti_label_dir中的文件复制到label_dir下，并递增文件名
#   print(f"start to copy copy_and_rename_files,max_suffix:{max_suffix}")
#     # label
#     dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/label_3"
#     kitti_dir = "/data2/monoDETR/code/train_0/data/data_0/KITTIDataset/training/label_2"
#     copy_and_rename_files(kitti_dir, dst_dir, max_suffix)
    # calib
    # max_suffix = 68889
    # dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/calib_3"
    # kitti_dir = "/data2/monoDETR/code/train_0/data/data_half_train/KITTIDataset/training/calib"
    # copy_and_rename_files(kitti_dir, dst_dir, max_suffix)
#     # image
#     dst_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/image_3"
#     kitti_dir = "/data2/monoDETR/code/train_0/data/data_0/KITTIDataset/training/image_2"
#     copy_and_rename_files(kitti_dir, dst_dir, max_suffix)
    generate_imageset()
    
