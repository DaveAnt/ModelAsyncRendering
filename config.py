import os
import bpy
import math

task_num = 1
batch_num = 1
post_process = True
texture_size = [1024,1024]
render_mode = {
    'Inextrinsics' : False,
    'FinalColor' : True,
    'Normal' : False,
    'Depth' : True,
    'Albedo' : False,
    'FaceIndex' : False,
    'Roughness' : False,
    'Metallic' : False,
    'Mask' : False
}

#保存FaceIndex时。使用255进制
bin_val = 255
in_bin_val = bin_val + 1

output_path = "C:/Users/kingding/Desktop/lgame_render_output/"
input_path = "C:/Users/kingding/Documents/WXWork/1688857177314719/Cache/File/2025-01/"
prj_path = os.path.dirname(os.path.abspath(__file__))

def cam_angles(elevation):#相机初始角度
    cam_angles = []
    for z in range(0, 360, 45):
        for x in range(0, 360, 45):
            cam_angles.append([x,0,z])
    return cam_angles

def scene_infos():
    infos = []
    for _ in range(1):
        info = {}
        info['fov'] = 60
        info['distance'] = 1.3
        info['elevation'] = [0,0,0]
        infos.append(info)
    return infos

def handle_scene_state(info):
    bpy.context.scene.camera.location = (0, -info['distance'], 0)
    bpy.context.scene.camera.data.angle = math.radians(info['fov'])

#判断模型是否满足要求
def check_model_state(model,file_path):
    # materials = []
    # scene = bpy.context.scene
    # for obj in scene.objects:
    #     if obj.material_slots:
    #         for slot in obj.material_slots:
    #             if slot.material:
    #                 materials.append(slot.material)
    # bsdf_nodes = []
    # for mat in materials:
    #     nodes = mat.node_tree.nodes
    #     for node in nodes:
    #         if node.type == 'BSDF_PRINCIPLED':
    #             bsdf_nodes.append(node)
    # for i in range(len(bsdf_nodes)):
    #     albedo_input_link = bsdf_nodes[i].inputs['Base Color'].links[0] if bsdf_nodes[i].inputs['Base Color'].links else None
    #     if albedo_input_link == None:
    #         with open('log.txt', 'a') as file:
    #             file.write(f'{file_path}:Albedo no exist!\n')
    #         return False
        
    #     roughness_input_link = bsdf_nodes[i].inputs['Roughness'].links[0] if bsdf_nodes[i].inputs['Roughness'].links else None
    #     if roughness_input_link == None:
    #         with open('log.txt', 'a') as file:
    #             file.write(f'{file_path}:Roughness no exist!\n')
    #         return False
        
    #     metallic_input_link = bsdf_nodes[i].inputs['Metallic'].links[0] if bsdf_nodes[i].inputs['Metallic'].links else None
    #     if metallic_input_link == None:
    #         with open('log.txt', 'a') as file:
    #             file.write(f'{file_path}:Metallic no exist!\n')
    #         return False
    return True

#判断保存状态是否正确（中断后重新运行）
def check_save_state(root_path):
    return False