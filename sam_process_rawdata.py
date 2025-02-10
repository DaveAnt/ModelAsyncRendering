import os
import Imath
import OpenEXR
from PIL import Image
import numpy as np
import argparse
import mathutils
import config

def get_format_files(path, format='.exr'):
    exr_files = []
    for file_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_name)):
            exr_files.extend(get_format_files(os.path.join(path, file_name)))
        elif file_name.endswith(format):
            exr_files.append(os.path.join(path, file_name))
    return exr_files

def post_process_mask(file_path):
    save_path = os.path.dirname(os.path.dirname(file_path))
    file_result = os.path.basename(file_path).split('.')  # 获取文件格式
    file_name,file_format = file_result[0],file_result[-1]

    # 打开PNG文件
    img = Image.open(file_path).convert("RGBA")
    alpha_channel = img.split()[3]
    alpha_channel.save(os.path.join(save_path,f'mask/{file_name}.png'),'PNG')

def post_process_faceid(file_path):
    save_path = os.path.dirname(file_path)
    file_result = os.path.basename(file_path).split('.')  # 获取文件格式
    file_name,file_format = file_result[0],file_result[-1]

    exr_file = OpenEXR.InputFile(file_path)
    exr_header = exr_file.header()

    # 获取EXR文件的尺寸
    dw = exr_header['dataWindow']
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    
    # 定义通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = list(reversed(exr_header['channels'].keys()))

    # 读取每个通道的数据
    data = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in channels]
    
    # 重新调整数据形状
    data = [d.reshape(height, width) for d in data]
    
    # 合并通道数据
    img = np.stack(data, axis=-1)
    
    # 对每个像素进行转换
    face_ids = []
    for i in range(height):
        face_ids_row = []
        for j in range(width):
            r, g, b, a = img[i, j]
            if a != 0:
                value = round(round(r * config.bin_val) + round(g * config.bin_val) * config.in_bin_val + round(b * config.bin_val) * config.in_bin_val * config.in_bin_val)
                face_ids_row.append(value)
            else:
                face_ids_row.append(-1)
        face_ids.append(np.array(face_ids_row))
    np.save(os.path.join(save_path,f"{file_name}.npy"), np.array(face_ids))

def post_process_normal(file_path,camera_mat):
    save_path = os.path.dirname(file_path)
    file_result = os.path.basename(file_path).split('.')  # 获取文件格式
    file_name,file_format = file_result[0],file_result[-1]

    # 打开EXR文件
    exr_file = OpenEXR.InputFile(file_path)
    exr_header = exr_file.header()

    # 获取EXR文件的尺寸
    dw = exr_header['dataWindow']
    
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    
    # 定义通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = list(reversed(exr_header['channels'].keys()))
    # 读取每个通道的数据
    data = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in channels]
    
    # 重新调整数据形状
    data = [d.reshape(height, width) for d in data]
    
    # 合并通道数据
    img = np.stack(data, axis=-1)
    
    # 将RGB转换为XYZ并进行归一化处理
    def rgb_to_xyz(r, g, b):
        x = (r - 0.5) * 2.0
        y = (g - 0.5) * 2.0
        z = (b - 0.5) * 2.0
        return x, y, z
    
    def xyz_to_rgb(x, y, z):
        r = (x / 2.0) + 0.5
        g = (y / 2.0) + 0.5
        b = (z / 2.0) + 0.5
        return r, g, b
    
    # 定义旋转角度
    # angle_x = math.radians(90)  # 将 90 度转换为弧度
    # angle_z = math.radians(180)  # 将 180 度转换为弧度

    # # 构建绕 X 轴旋转的矩阵
    # rotation_matrix_x = mathutils.Matrix.Rotation(angle_x, 4, 'X')

    # # 构建绕 Z 轴旋转的矩阵
    # rotation_matrix_z = mathutils.Matrix.Rotation(angle_z, 4, 'Z')

    # # 组合旋转矩阵：先绕 X 轴旋转，再绕 Z 轴旋转
    # combined_rotation_matrix = rotation_matrix_z @ rotation_matrix_x

    # 对每个像素进行转换
    for i in range(height):
        for j in range(width):
            x, y, z = img[i, j]
            if x == 0 and y == 0 and z == 0:
                img[i, j] = [0, 0, 0]
            else:
                norm = camera_mat @ mathutils.Vector((x, y, z))
                r, g, b = xyz_to_rgb(norm[0],norm[1],norm[2])

                # r, g, b = xyz_to_rgb(x,y,z)
                img[i, j] = [norm[0], norm[1], norm[2]]
                #img[i, j] = [r, g, b]

    # 将浮点数转换为8位整数
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # 创建Pillow图像对象
    img = Image.fromarray(img)
    # 保存为PNG文件
    img.save(os.path.join(save_path,f'{file_name}.png'),'PNG')

def post_process_depth(file_path):
    save_path = os.path.dirname(file_path)
    file_result = os.path.basename(file_path).split('.')  # 获取文件格式
    file_name,file_format = file_result[0],file_result[-1]

    exr_file = OpenEXR.InputFile(file_path)
    # 获取文件头信息
    exr_header = exr_file.header()

    dw = exr_header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    channels = list(reversed(exr_header['channels'].keys()))
    depth_channel = exr_file.channel(channels[0], Imath.PixelType(Imath.PixelType.FLOAT))
    depth_data = np.frombuffer(depth_channel, dtype=np.float32)
    #depth_data.shape = (size[0], size[1])  # 设置正确的形状

    max_val = np.max(depth_data)
    min_val = np.min(depth_data)
    mout_val = np.max(depth_data[depth_data != max_val])
    len_val = mout_val - min_val

    # with open(os.path.join(folder_path,f'{file_name}.pfm'), "wb") as file:
    #     header = "Pf\n{} {}\n-1.000000\n".format(size[0], size[1])
    #     file.write(header.encode())

    #     for i in range(len(depth_data)-1, -1, -1):
    #         depth = depth_data[i]
    #         depth_norm = 1 if depth >= mout_val else (depth - min_val)/len_val
    #         file.write(struct.pack("f", depth_norm))
    #     file.close()

    depth_image = Image.new("L", (size[0], size[1]))
    for i in range(len(depth_data)):
        row = int(i % size[0])
        col = int(i / size[0])

        depth = depth_data[i]
        depth_norm = 1 if depth >= mout_val else (depth - min_val)/len_val
        depth_norm = 1 - depth_norm

        R = depth_norm * 255
        # G = (depth_norm * 65025) % 256
        # B = (depth_norm * 16581375) % 256
        depth_image.putpixel((row, col), int(R))

    depth_image.save(os.path.join(save_path,f'{file_name}.png'),'PNG')
    print(f'{file_path} finish!')

def post_process_final(file_path):
    save_path = os.path.dirname(file_path)
    file_result = os.path.basename(file_path).split('.')  # 获取文件格式
    file_name,file_format = file_result[0],file_result[-1]

    try:
        # 不能确定正确执行的代码
        img = Image.open(file_path)

        # 创建一个白色背景的图片
        new_img = Image.new("RGB", img.size, (255, 255, 255))

        # 将透明背景的图片粘贴到白色背景上
        new_img.paste(img, (0, 0), img)

        # 保存新图片
        new_img.save(os.path.join(save_path,f'white_{file_name}.png'))

        print(f'{file_path} finish!')
    except:
        print('无法mask')


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Input PostProcessing Params.')

    # 添加参数
    parser.add_argument("--exr_folder",
                        type=str,
                        default="",
                        help="output weight file path")
    parser.add_argument("--png_folder",
                    type=str,
                    default="",
                    help="output weight file path")
    
    args = parser.parse_args()
    png_file_paths = get_format_files(args.png_folder,'.png')
    folder_path = os.path.dirname(args.png_folder)
    folder_name = os.path.basename(args.png_folder)
    save_path = os.path.join(folder_path,'post_'+folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file_path in png_file_paths:
        post_process_final(file_path)
    
    exr_path = args.exr_folder
    exr_file_paths = get_format_files(exr_path)
    folder_path = os.path.dirname(exr_path)
    folder_name = os.path.basename(exr_path)
    save_path = os.path.join(folder_path, 'post_'+folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_path in exr_file_paths:
        post_process_depth(file_path)

if __name__ == '__main__':
    main()