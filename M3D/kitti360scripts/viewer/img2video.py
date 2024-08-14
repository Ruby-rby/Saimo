import cv2
import os
from moviepy.editor import VideoFileClip

def images_to_video(image_folder, video_path, video_name, fps=30):
    # 获取所有图片文件名，并按顺序排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    # images = [img for img in os.listdir(image_folder) if img.endswith("_(255, 0, 255)_box3d.png")]
    images.sort()  # 确保图片按顺序排列

    # 从第一张图片读取帧的宽高
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    video = cv2.VideoWriter(video_path+video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放VideoWriter对象
    video.release()
    cv2.destroyAllWindows()
    
    
    # 转gif
    mp4_path = os.path.join(video_path, video_name)  # 输入 MP4 文件路径
    gif_path = os.path.join(video_path, video_name.replace("mp4", "gif"))  # 输出 GIF 文件路径

    # 将整个视频转换为 GIF
    convert_mp4_to_gif(mp4_path, gif_path)

    # 将视频的一部分转换为 GIF，并缩小 50%
    # convert_mp4_to_gif(mp4_path, 'output_resized.gif', start_time=5, end_time=10, resize=0.5)
    
def convert_mp4_to_gif(mp4_path, gif_path, start_time=None, end_time=None, resize=None):
    """
    将 MP4 转换为 GIF.

    :param mp4_path: 输入的 MP4 文件路径
    :param gif_path: 输出的 GIF 文件路径
    :param start_time: 起始时间（秒），默认为 None
    :param end_time: 结束时间（秒），默认为 None
    :param resize: 调整 GIF 大小的比例（例如 0.5 表示缩小 50%），默认为 None
    """
    # 加载视频文件
    clip = VideoFileClip(mp4_path)

    # # 裁剪视频时长
    # if start_time is not None or end_time is not None:
    #     clip = clip.subclip(start_time, end_time)

    # 调整视频大小
    if resize is not None:
        clip = clip.resize(resize)
    
    # # 设置目标帧率（如：10fps）
    clip = clip.set_fps(10)

    # 将视频转换为 GIF
    clip.write_gif(gif_path, fps=10)

# 使用示例
# dir
seq = "2013_05_28_drive_0010_sync"
image_folder = f'/data2/KITTI-360/data_2d_raw/{seq}/image_00/data_rect'  # 替换为图片所在文件夹的路径
video_path = "/data2/KITTI-360/gif/"
video_name = f'{seq}.mp4'  # 输出视频的文件名
fps = 5  # 每秒帧数

images_to_video(image_folder, video_path, video_name, fps)
