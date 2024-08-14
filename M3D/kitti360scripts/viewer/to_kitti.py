import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
import warnings
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, "/data2/KITTI-360/kitti360Scripts-master/")
# annotation helper
from kitti360scripts.helpers.annotation  import Annotation3D, Annotation3DPly, global2local
from kitti360scripts.helpers.project     import Camera, CameraPerspective
from kitti360scripts.helpers.labels      import name2label, id2label, kittiId2label

# sys.path.append("/data2/kitti360/SeaBird-main/PanopticBEV")
# from panoptic_bev.helpers.seman_helper import metric_to_bev_px, get_obj_level



def get_objects_from_label(label_file):
    """根据标注数据获取检测对象"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    # print("lines:", lines)
    objects = [Object3d(line) for line in lines if line.split(" ")[0] in "Car"]
    return objects


class Object3d(object):
    
    BEV_SHAPE = None
    MIN_XZ = None
    
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        # self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.score = float(label[-1])
        self.level_str = None
        # self.level = self.get_obj_level()


    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4


    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        # y_corners = [0, 0, 0, 0, -h, -h, -h, -h] # zltst
        y_corners = [ -h, -h, -h, -h, 0, 0, 0, 0]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        corners3d[:,1] *= -1 # zltst 0709
        return corners3d


    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d


    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str


    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str



def draw_projected_box3d(image, corners3d, color=(255, 255, 255), thickness=1):
    ''' Draw 3d bounding box in image
    input:
        image: RGB image
        corners3d: (8,3) array of vertices (in image plane) for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    # self.lines = [[0,4],[1,5],[3,7],[2,6],
                      [0,1],[1,2],[2,3],[3,0],
                      [5,4],[4,7],[7,6],[6,5]]

            1 -------- 0
           /|         /|
          3 -------- 2 .
          | |        | |
          . 4 -------- 5
          |/         |/
          6 -------- 7
    # self.lines = [[0,5],[1,4],[2,7],[3,6],
                      [0,1],[1,3],[3,2],[2,0],
                      [4,5],[5,7],[7,6],[6,4]]
    '''

    corners3d = corners3d.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        # print("1: ", (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]))
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        # print("2:", (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]))
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        i, j = k, k + 4
        # print("3:", (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]))
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv2.LINE_AA)
        
    return image



def read_lines(path, strip= True):
    with open(path) as f:
        lines = f.readlines()

    if strip:
        # you may also want to remove whitespace characters like `\n` at the end of each line
        lines = [x.rstrip('\n') for x in lines]

    return lines

def get_frame_to_kitti_index_dict():
    """{"2013_05_28_drive_0004_sync-0000009808": "000000", "2013_05_28_drive_0006_sync-0000008584":"068889"}"""
    inp_id_list_path = "/data2/KITTI-360/Ext_ImageSets/org_trainval_det_clean.txt"
    out_id_list_path = "/data2/KITTI-360/Ext_ImageSets/trainval_det.txt"
    inp_id_list = read_lines(inp_id_list_path)
    out_id_list = read_lines(out_id_list_path)
    
    frame2idx = {}
    for index, val in enumerate(inp_id_list):
        key = val.replace(";", "-")
        frame2idx[key] = out_id_list[index]
    print("get_frame_to_kitti_index_dict", sorted(frame2idx))
    return frame2idx

def get_3d_to_2d_from_360():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sequence', type=int, default=3, help='The sequence to visualize')
    args = parser.parse_args()
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = "/data2/KITTI-360/"
    # 3D bbox
    seq = args.sequence
    sequence = '2013_05_28_drive_%04d_sync' % seq
    
    frame2idx = get_frame_to_kitti_index_dict() 
    
    label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
    annotation3D = Annotation3D(label3DBboxPath, sequence)  # self.objects[globalId][obj.timestamp] = obj (KITTI360Bbox3D)
    # perspective
    
    cam_id = 0
    camera = CameraPerspective(kitti360Path, sequence, cam_id)
    
    data_obj = {}  # data_obj = {"frame_id": {"3d_points": [], "2d_lines": []}, "3d_vertices": []}
    
    # loop over frames
    for frame in camera.frames:
        frame = int(frame)
        

        points = []
        depths = []
        
        data_obj[frame] = {"2d_lines": []}
            
        # 遍历data_3d_bboxes的一个xml标注文件
        for k,v in annotation3D.objects.items():  # obj = annotation3D.objects.26001.-1
            # for obj in v.values():
            #     lines=np.array(obj.lines)
            #     vertices=obj.vertices
            if len(v.keys())==1 and (-1 in v.keys()): # show static only
                dynamic = -1
            elif frame in v.keys():  # 动态
                dynamic = frame
            else:
                continue
            obj3d = v[dynamic]
            
            if not id2label[obj3d.semanticId].name=='car': # only show car
                continue
            if frame not in range(obj3d.start_frame, obj3d.end_frame+1):
                continue
            
            print(f"---------{obj3d.annotationId}---------")
            print(f"name: {obj3d.name}")
            print(f"dynamic: {1 if dynamic>0 else 0}")
            print(f"timestamp: {obj3d.timestamp}")
            print(f"frame: {frame} in {obj3d.start_frame}-{obj3d.end_frame}")
            
            camera(obj3d, frame)
            vertices = np.asarray(obj3d.vertices_proj).T
            points.append(np.asarray(obj3d.vertices_proj).T)
            depths.append(np.asarray(obj3d.vertices_depth))
            
            print(f"[{frame}] vertices: {len(vertices)}, {vertices}")
            
            lines = []
            for line in obj3d.lines:
                # self.lines = [[0,5],[1,4],[2,7],[3,6], 
                #     [0,1],[1,3],[3,2],[2,0],
                #     [4,5],[5,7],[7,6],[6,4]]
                
                # print(f"obj3d.vertices_depth: {obj3d.vertices_depth}")
                
                # # 跳过两个顶点深度都为负值的线条
                # if obj3d.vertices_depth[line[0]]<0 and obj3d.vertices_depth[line[1]]<0:
                #     continue
                # # 处理其中一个顶点深度为负值的线条
                # elif obj3d.vertices_depth[line[0]]<0 or obj3d.vertices_depth[line[1]]<0:
                #     # 插值计算新的顶点位置
                #     uv_v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                #     uv, d = camera.project_vertices(np.asarray(uv_v), frame)
                        
                #     d[d<0] = 1e+6
                #     vidx = line[0] if obj3d.vertices_depth[line[0]] < 0 else line[1]
                #     obj3d.vertices_proj[0][vidx] = uv[0][np.argmin(d)]
                #     obj3d.vertices_proj[1][vidx] = uv[1][np.argmin(d)]
                # 记录一条线段（cam2world -> world2cam -> cam2image）
                line_1 = (
                    obj3d.vertices_proj[0][line[0]],
                    obj3d.vertices_proj[1][line[0]],
                    obj3d.vertices_proj[0][line[1]],
                    obj3d.vertices_proj[1][line[1]])
                lines.append(line_1)
            
            if lines:
                # 记录3DBBox的12条边
                print(f"[{frame}] len lines: {len(lines)}, {lines}")
                data_obj[frame]["2d_lines"].append(lines)
                # data_obj[frame]["3d_vertices"].append(obj3d.vertices)
    
    print(f"data_obj: {data_obj}")
   
    output_dir = os.path.join(kitti360Path, 'outputs_0710')
    
    # loop over data_obj, output image with rectangle
    for frame in data_obj.keys():
        
        # 限制frame只要3的测试文件
        # if frame not in [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 139, 140, 141, 143, 145, 148, 151, 154, 157, 159, 161, 163, 165, 167, 179, 180, 181, 182, 183, 184, 185, 186, 187, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 480, 481, 482, 483, 484, 485, 486, 487, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 671, 672, 673, 674, 675, 676, 677, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 804, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 858, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 979, 980, 981, 982, 983]:
        #     print(f"{sequence}-{frame:010d}, not available")
        #     continue
        
        # perspective
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect', '%010d.png'%frame)
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        obj_lines = data_obj[frame]["2d_lines"]
        
        if not obj_lines:
            continue
         
        try:
            print(image_file, image.shape)
            print(f"lines: {len(lines)}, {lines}")
            out = True
            
            in_img = 24
            for lines in obj_lines:
                for (x1, y1, x2, y2) in lines:
                    start_point = (x1, y1)
                    end_point = (x2, y2)
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
                    # # 判断线段是否在图像内
                    # if start_point[0] not in range(0, 1409) or start_point[1] not in range(0, 377):  # 376, 1408
                    #     print(f"start_point: {start_point}")
                    #     in_img -= 1
                    # if end_point[0] not in range(0, 1409) or end_point[1] not in range(0, 377):
                    #     print(f"end_point: {end_point}")
                    #     in_img -= 1
            # if in_img >= 12:
            #     out = False
            # # 如果没有3D框，不保存图像
            # if out:
            #     print(f"{frame} has no object")
            #     continue
            
        
        except Exception as e:
            print(e, traceback.format_exc())
        
        
        # 获取图像尺寸
        height, width, _ = image.shape

        # 设置帧号文本
        text = f"Frame {frame:010d}"
        
        # 获取文本尺寸
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

        # 计算文本位置 (右上角)
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10

        # 在图像上添加文本
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
 
        # 保存图像
        output_image_file = os.path.join(output_dir, f'{frame:010d}_bbox_o.png')
        cv2.imwrite(output_image_file, image)
        print(output_image_file)


def read_csv(path, delimiter= " ", ignore_warnings= False, use_pandas= False):
    try:
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_pandas:
                    data = pd.read_csv(path, delimiter= delimiter, header=None).values
                else:
                    data = np.genfromtxt(path, delimiter= delimiter)
        else:
            if use_pandas:
                data = pd.read_csv(path, delimiter=delimiter, header=None).values
            else:
                data = np.genfromtxt(path, delimiter=delimiter)
    except Exception as e:
        print(e, traceback.format_exc()) 
        data = None


def read_file(path, delimiter=" "):
    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                tokens = line.strip().split(delimiter)
                data.append(tokens)        
    except Exception as e:
        print(e, traceback.format_exc())
    data = np.array(data) if data else None
    return data
    


def get_3d_to_2d_from_ext_label():
    """根据外部标注信息，确定3d框"""
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sequence', type=int, default=3, help='The sequence to visualize')
    args = parser.parse_args()
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = "/data2/KITTI-360/"
    
    # 3D bbox
    seq = args.sequence
    sequence = '2013_05_28_drive_%04d_sync' % seq
    print(f"sequence: {sequence}")
    
    frame2idx = get_frame_to_kitti_index_dict()
    
    
    label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
    # annotation3D = Annotation3D(label3DBboxPath, sequence)  # self.objects[globalId][obj.timestamp] = obj (KITTI360Bbox3D)
    # perspective
    
    cam_id = 0
    camera = CameraPerspective(kitti360Path, sequence, cam_id)

    # 仅输出cam2world
    return
    data_obj = {}  # data_obj = {"frame_id": {"3d_points": [], "2d_lines": []}, "3d_vertices": []}
    
    # loop over frames
    for frame in camera.frames:
        frame = int(frame)

        points = []
        depths = []
        
        data_obj[frame] = {"2d_lines": []}
            
        # 遍历data_3d_bboxes的一个xml标注文件
        # for k,v in annotation3D.objects.items():  # obj = annotation3D.objects.26001.-1
        
    
        kitti_index = frame2idx.get(f"{sequence}-{frame:010d}", None)
        if kitti_index is None:
            print(f"{sequence}-{frame:010d}, not available")
            continue
        label_file_path = f"{kitti360Path}/Ext_Label/{kitti_index}.txt"

        objects = get_objects_from_label(label_file_path)
        
        corners_list = []
        
        # 初始化为一个大的正数和小的负数，便于找到最小和最大值
        min_x = float('inf')
        max_x = float('-inf')
        min_z = float('inf')
        max_z = float('-inf')
        
        
        # img2 = np.copy(img)  # for 3d bbox
        
        for object in objects:
            print(f"[{frame}] box2d:{object.box2d}, h:{object.h}, w:{object.w}, l:{object.l}, pos:{object.pos}, alpha:{object.alpha}")
            
            corners3d = object.generate_corners3d()

            if not corners3d.any():
                print("no corners3d!")
            
            corners_list.append(corners3d)
            # print("corners3d:", np.array(corners3d).shape, corners3d)
            new_corners = corners3d.T
            # print("new_corners:", new_corners.shape, new_corners)
            u, v, depth = camera.cam2image(new_corners)
            
            object.lines = [[0,4],[1,5],[3,7],[2,6],
                      [0,1],[1,2],[2,3],[3,0],
                      [5,4],[4,7],[7,6],[6,5]]
            bbox_lines = []
            for l_i in object.lines:
                line_1 = (u[l_i[0]],v[l_i[0]], u[l_i[1]],v[l_i[1]])
                bbox_lines.append(line_1)
            # print(f"bbox_lines: {bbox_lines}")
            
            
            if bbox_lines:
                # 记录3DBBox的12条边
                # print(f"[{frame}] len lines: {len(bbox_lines)}, {bbox_lines}")
                data_obj[frame]["2d_lines"].append(bbox_lines)
    
    # print(f"data_obj: {data_obj}")
   
    output_dir = os.path.join(kitti360Path, 'outputs_new')
    
    # loop over data_obj, output image with rectangle
    for frame in data_obj.keys():
        # perspective
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect', '%010d.png'%frame)
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        obj_lines = data_obj[frame]["2d_lines"]
        
        if not obj_lines:
            continue
         
        try:
            print(image_file, image.shape)
            
            for lines in obj_lines: 
                for (x1, y1, x2, y2) in lines:
                    start_point = (x1, y1)
                    end_point = (x2, y2)
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
                    # 判断线段是否在图像内
                    if start_point[0] not in range(0, 1409) or start_point[1] not in range(0, 377):  # 376, 1408
                        print(f"start_point: {start_point}")
                    if end_point[0] not in range(0, 1409) or end_point[1] not in range(0, 377):
                        print(f"end_point: {end_point}")
        except Exception as e:
            print(e, traceback.format_exc())
        
        # 获取图像尺寸
        height, width, _ = image.shape

        # 设置帧号文本
        text = f"Frame {frame:010d}"
        
        # 获取文本尺寸
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

        # 计算文本位置 (右上角)
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10

        # 在图像上添加文本
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # 保存图像
        output_image_file = os.path.join(output_dir, f'{frame:010d}_bbox.png')
        cv2.imwrite(output_image_file, image)
        print(output_image_file)

if __name__ == '__main__':
    kitti360Path = "/data2/KITTI-360/"
    get_3d_to_2d_from_ext_label()
    
    # get_3d_to_2d_from_360()
    print("ok")

