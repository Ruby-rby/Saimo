import os
import sys
import numpy as np 

def check():
    label_dir = "/data2/monodetr-0722/train_1/data/data_0/KITTIDataset/training/label_3"
    for root, dirs, files in os.walk(label_dir):  
        for file in files:  
            # 检查文件是否以.txt结尾  
            if file.endswith('.txt'):  
                # 构造文件的完整路径  
                file_path = os.path.join(root, file)  
                
                # 打开文件并逐行读取  
                
                with open(file_path, 'r', encoding='utf-8') as f:  # 使用utf-8编码，根据你的文件实际情况调整  
                    for line in f:  
                        # 处理每一行内容  
                        print(line.strip())  # 去除行尾的换行符并打印
                        
                        bbox = [eval(i) for i in line.split()[4: 8]]
                        box_np = np.array(bbox)
                            
                        print(f"{len(line)}|{bbox}")
                        assert (box_np[2:] >= box_np[:2]).all()  
                        # os._exit(0)
    # idx_list = [filename.split(".")[0] for filename in os.listdir(label_dir) if filename.endswith(".txt")]
    sys.exit()
        # score_list = [i[16] for i in output_list if len(i)>16]
        # print(f"score_list: {score_list}")
        # scores = np.array([eval(i[16]) for i in output_list if len(i)>16])  # 示例置信度分数
        # labels = np.array([i[0] for i in output_list])  # 示例标签
        # label_map = {1: 'Car', 'Car': 'Car', "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", 
        #                 "Van": "Van", "Person_sitting": "Person_sitting", "Truck": "Truck"}  # 标签映射
    
check()