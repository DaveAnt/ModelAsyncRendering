## ModelAsyncRendering 简介
通过blend后台多进程对模型进行自定义视角的渲染，支持实物、法线、深度、Albedo、粗糙、金属度、遮罩、面索引转换成255进制、相机内外参矩阵的数据导出，作为训练图形相关AI的数据集（支持Win、Linux环境运行）。<br>
安装requirements.txt，配置好config.py，最后运行start_render_tasks.py即可。

## 相关配置文件
config.py是渲染相关的配置参数：
- "task_num"：渲染进程个数
- "batch_num": 单个渲染进程执行渲染任务个数（进程结束会销毁，重启新进程，相当于值越小，gc越大，适用于低内存电脑）
- "post_process" : 是否开启数据后处理
- "render_mode"
    - 'Inextrinsics' : 是否保存内外参数矩阵
    - 'FinalColor' : 是否保存模型实际渲染图
    - 'Normal' : 是否保存模型法线图
    - 'Depth' : 是否保存模型深度图
    - 'FaceIndex' : 是否保存模型面索引（255进制方式）
    - 'Roughness' : 是否保存模型粗糙值图
    - 'Metallic' : 是否保存模型金属度图
    - 'Mask' : 是否保存模型遮罩图
- "input_path" : 渲染的模型根路径
- "output_path" : 渲染结果的保存路径
- "texture_size" : 实际纹理大小
- "cam_angles" : 自定义的渲染角度
- "fov" : 相机fov值
