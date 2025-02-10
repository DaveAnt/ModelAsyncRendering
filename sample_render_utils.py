import os
import math
import json
import sys
import bpy
import bmesh
import config
import mathutils
import numpy as np
import sam_process_rawdata

def get_mesh_objects(obj):
    mesh_objects = []
    if obj.type == 'MESH':
        mesh_objects.append(obj)
    if hasattr(obj, 'children'):
        for child in obj.children:
            mesh_objects.extend(get_mesh_objects(child))
    return mesh_objects

def get_combined_bbox(mesh_objects):
    if not mesh_objects:
        return None

    # 初始化包围盒最小和最大值
    combined_bbox_min = [float('inf')] * 3
    combined_bbox_max = [float('-inf')] * 3

    # 遍历所有网格对象并更新包围盒最小和最大值
    for obj in mesh_objects:
        mesh = obj.data
        for vertex in mesh.vertices:
            world_vertex = obj.matrix_world @ vertex.co
            combined_bbox_min = [min(a, b) for a, b in zip(combined_bbox_min, world_vertex)]
            combined_bbox_max = [max(a, b) for a, b in zip(combined_bbox_max, world_vertex)]

    return mathutils.Vector(combined_bbox_min), mathutils.Vector(combined_bbox_max)

def select_all_childs(root):
    root.select_set(True)
    for obj in root.children:
        obj.select_set(True)
        if obj.hide_viewport:
            obj.hide_viewport = False
        select_all_childs(obj)

def select_model_root():
    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    model_sets = bpy.context.view_layer.objects.active
    model_sets.name = 'ModelSets'
    
    for obj in bpy.data.objects:
        if (obj.parent is None) and (obj.name != 'ModelPoint') and (obj.name != 'ModelSets'):
            obj.parent = model_sets
    return model_sets

def save_inextrinsics_matrix(output_path):
    camera = bpy.context.scene.camera

    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    # 获取相机的焦距和传感器大小
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height

    # 计算内参矩阵
    fx = (focal_length / sensor_width) * width
    fy = (focal_length / sensor_height) * height
    cx = width / 2
    cy = height / 2

    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

    location, rotation = camera.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = mathutils.Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))

    with open(output_path, 'w') as f:
        for row in RT:
            f.write("%f %f %f %f\n" % (row[0], row[1], row[2], row[3]))

def encode_index_to_rgba(index):
    r = index % config.in_bin_val / config.bin_val
    g = (index // config.in_bin_val) % config.in_bin_val / config.bin_val
    b = (index // (config.in_bin_val * config.in_bin_val)) % config.in_bin_val / config.bin_val
    index = round(r * config.bin_val) + round(g * config.bin_val) * config.in_bin_val + round(b * config.bin_val) * config.in_bin_val * config.in_bin_val
    return (r , g , b , 1)

def create_face_index_material():
    scene = bpy.context.scene
    for obj in scene.objects:
        if obj.type == 'MESH':

            mat = bpy.data.materials.new(name="FaceIDMat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            for node in nodes:
                nodes.remove(node)
            
            emission_node = nodes.new(type='ShaderNodeEmission')
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            attribute_node = nodes.new(type='ShaderNodeAttribute')

            attribute_node.attribute_name = "Col"
            
            #links.new(attribute_node.outputs['Color'], emission_node.inputs['Color'])

            links.new(attribute_node.outputs['Color'], output_node.inputs['Surface'])
            output_node.target = 'EEVEE'
            
            if mat.name not in [m.name for m in obj.data.materials]:
                obj.data.materials.append(mat)

            bm = bmesh.new()
            bm.from_mesh(obj.data)

            color_layer = bm.loops.layers.color.get("Col")
            if color_layer is None:
                color_layer = bm.loops.layers.color.new("Col")

            for face in bm.faces:
                rgba = encode_index_to_rgba(face.index)
                for loop in face.loops:
                    loop[color_layer] = rgba

                face.material_index = obj.data.materials.find(mat.name)

            bm.to_mesh(obj.data)
            bm.free()

def fetch_save_path(root_path, total_num, index):
    if total_num > 1:
        return os.path.join(root_path,index)
    else:
        return root_path

if __name__ == '__main__':
    with open("./renders.json", 'r') as file:
        model_files = json.load(file)
        model_count = len(model_files)

        index = int(sys.argv[-2])
        batch_num = int(sys.argv[-1])
        end_index = min(model_count, index + batch_num)

        input_path = config.input_path 
        output_path = config.output_path
        enable_post = config.post_process

        enable_poses = config.render_mode['Inextrinsics']
        enable_final = config.render_mode["FinalColor"]
        enable_faceid = config.render_mode['FaceIndex']
        enable_depth = config.render_mode['Depth']
        enable_normal = config.render_mode['Normal']
        enable_albedo = config.render_mode['Albedo']
        enable_roughness = config.render_mode['Roughness']
        enable_metallic = config.render_mode['Metallic']
        enable_mask = config.render_mode['Mask']
        
        # 选择所有对象
        bpy.ops.object.select_all(action='SELECT')
        # 删除所有对象
        bpy.ops.object.delete()

        bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        model_point = bpy.context.view_layer.objects.active
        model_point.name = 'ModelPoint'

        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -1.3, 0), rotation=(math.radians(90), 0, 0), scale=(1, 1, 1))
        camera = bpy.context.view_layer.objects.active
        bpy.context.scene.camera = camera
        camera.data.lens_unit = 'FOV'
        camera.parent = model_point
        camera.name = 'Camera'

        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', rotation=(0, 0, 0), location=(0, 0, 0), scale=(1, 1, 1))
        light = bpy.context.view_layer.objects.active
        light.parent = camera
        light.name = 'Light'

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filter_size = 0
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.exr_codec = 'ZIP'
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_depth = '16'
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1
        bpy.context.object.data.cycles.use_multiple_importance_sampling = False
        bpy.context.object.data.cycles.cast_shadow = False
        bpy.context.object.data.cycles.max_bounces = 0
        bpy.context.object.data.energy = 8
        
        bpy.context.scene.cycles.max_bounces = 6
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transmission_bounces = 2
        bpy.context.scene.cycles.volume_bounces = 0
        bpy.context.scene.cycles.transparent_max_bounces = 2

        bpy.context.scene.world.cycles_visibility.camera = True
        bpy.context.scene.world.cycles_visibility.diffuse = False
        bpy.context.scene.world.cycles_visibility.glossy = False
        bpy.context.scene.world.cycles_visibility.transmission = False
        bpy.context.scene.world.cycles_visibility.scatter = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.use_pass_direct = False

        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'

        bpy.context.scene.eevee.taa_render_samples = 1
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.cycles.adaptive_threshold = 0.1
        bpy.context.scene.cycles.use_denoising = False

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0
        bpy.context.scene.frame_step = 0

        bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

        bpy.context.scene.use_nodes = True
        node_tree = bpy.context.scene.node_tree

        # 清空合成节点
        for node in node_tree.nodes:
            node_tree.nodes.remove(node)

        render_layers = node_tree.nodes.new('CompositorNodeRLayers')
        if enable_depth:
            depth_file_output = node_tree.nodes.new(type="CompositorNodeOutputFile")
            depth_file_output.name = depth_file_output.label = 'Depth'
            node_tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
            depth_file_output.format.file_format = 'OPEN_EXR'
            depth_file_output.format.color_depth = '32'
            
        if enable_final:
            final_file_output = node_tree.nodes.new(type="CompositorNodeOutputFile")
            final_file_output.name = final_file_output.label = 'Final'
            node_tree.links.new(render_layers.outputs['Image'], final_file_output.inputs[0])
            final_file_output.format.color_mode = 'RGBA'
            final_file_output.format.file_format = 'PNG'
            final_file_output.format.color_depth = '8'

        if enable_normal:
            normal_file_output = node_tree.nodes.new(type="CompositorNodeOutputFile")
            node_tree.links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
            normal_file_output.name = normal_file_output.label = 'Normal'
            normal_file_output.format.file_format = 'OPEN_EXR'
            normal_file_output.format.color_mode = 'RGBA'

        # 设置渲染分辨率
        bpy.context.scene.render.resolution_x = config.texture_size[0]
        bpy.context.scene.render.resolution_y = config.texture_size[1]
        bpy.context.scene.render.resolution_percentage = 100

        for k in range(index, end_index):
            frame_num = 0
            file_path = os.path.join(input_path,model_files[k])
            file_result = os.path.basename(file_path).split('.')  # 获取文件格式
            file_name,file_format = file_result[0].replace(" ", "_"),file_result[-1]
            root_path = os.path.join(output_path,file_name)

            if config.check_save_state(root_path):
                continue

            if file_format == 'fbx':
                bpy.ops.import_scene.fbx(filepath=file_path)
            elif file_format == 'glb' or file_format == 'gltf':
                bpy.ops.import_scene.gltf(filepath=file_path)
            elif file_format == 'obj':
                bpy.ops.wm.obj_import(filepath=file_path)

            render_obj = select_model_root()
            bpy.context.view_layer.objects.active = render_obj
            mesh_objects = get_mesh_objects(render_obj)

            if not config.check_model_state(render_obj,file_path):
                continue

            materials = []
            scene = bpy.context.scene
            for obj in scene.objects:
                if obj.material_slots:
                    for slot in obj.material_slots:
                        if slot.material:
                            materials.append(slot.material)
            bsdf_nodes = []
            output_nodes = []
            for mat in materials:
                mat.blend_method = 'OPAQUE'
                nodes = mat.node_tree.nodes
                for node in nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        bsdf_nodes.append(node)
                    elif node.type == 'OUTPUT_MATERIAL':
                        output_nodes.append(node)

            # 刷新场景
            bpy.context.view_layer.update()
            # 获取所有网格对象的包围盒之和
            combined_bbox_min, combined_bbox_max = get_combined_bbox(mesh_objects)

            # 计算包围盒中心
            bbox_center = (combined_bbox_min + combined_bbox_max)/2
            bbox_size = (combined_bbox_max - combined_bbox_min)/2

            # 获取当前场景中的相机
            box_scale = 1 / max(combined_bbox_max - combined_bbox_min)
            render_obj.location = (-bbox_center+render_obj.location)*box_scale
            render_obj.scale = render_obj.scale * box_scale

            cam_poses = []
            scene_infos = config.scene_infos()
            infos_len = len(scene_infos)
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.render.filter_size = 0

            for k in range(infos_len):
                frame_num = 0
                k_string = str(k)
                save_path = fetch_save_path(root_path,infos_len,k_string)
                final_path = os.path.join(save_path,f"final")
                faceid_path = os.path.join(save_path,f"faceid")
                depth_path = os.path.join(save_path,f"depth")
                normal_path = os.path.join(save_path,f"normal")
                albedo_path = os.path.join(save_path,f"albedo")
                roughness_path = os.path.join(save_path,f"roughness")
                metallic_path = os.path.join(save_path,f"metallic")
                mask_path = os.path.join(save_path,f"mask")

                if enable_final and (not os.path.exists(final_path)):
                    os.makedirs(final_path)
                if enable_faceid and (not os.path.exists(faceid_path)):
                    os.makedirs(faceid_path)
                if enable_depth and (not os.path.exists(depth_path)):
                    os.makedirs(depth_path)
                if enable_normal and (not os.path.exists(normal_path)):
                    os.makedirs(normal_path)
                if enable_albedo and (not os.path.exists(albedo_path)):
                    os.makedirs(albedo_path)
                if enable_roughness and (not os.path.exists(roughness_path)):
                    os.makedirs(roughness_path)
                if enable_metallic and (not os.path.exists(metallic_path)):
                    os.makedirs(metallic_path)
                if enable_mask and (not os.path.exists(mask_path)):
                    os.makedirs(mask_path)

                handle_data = scene_infos[k]
                config.handle_scene_state(handle_data)
                cam_angles = config.cam_angles(handle_data['elevation'])

                for rot_value in cam_angles:
                    bpy.context.scene.frame_set(frame_num)
                    radians_value = [math.radians(deg) for deg in rot_value]
                    radians_value[0] = -radians_value[0]
                    model_point.rotation_euler = radians_value

                    for i in range(len(bsdf_nodes)):
                        links = materials[i].node_tree.links
                        links.new(bsdf_nodes[i].outputs['BSDF'], output_nodes[i].inputs['Surface'])

                    image_name = f"sample_" + '#' * len(str(frame_num))
                    bpy.context.scene.use_nodes = True

                    if enable_poses:
                        save_inextrinsics_matrix(os.path.join(final_path,f"sample_{frame_num}.txt"))
                    
                    if enable_final:
                        bpy.data.scenes["Scene"].node_tree.nodes["Final"].base_path = final_path#os.path.join(root_path,f"image_results_{k}")
                        bpy.data.scenes["Scene"].node_tree.nodes["Final"].file_slots[0].path = image_name

                    if enable_depth:
                        bpy.data.scenes["Scene"].node_tree.nodes["Depth"].base_path = depth_path#os.path.join(root_path,"depth_results")
                        bpy.data.scenes["Scene"].node_tree.nodes["Depth"].file_slots[0].path = image_name
                    
                    if enable_normal:
                        bpy.data.scenes["Scene"].node_tree.nodes["Normal"].base_path = normal_path#os.path.join(root_path,"depth_results")
                        bpy.data.scenes["Scene"].node_tree.nodes["Normal"].file_slots[0].path = image_name

                    # 渲染图像
                    bpy.ops.render.render()
                    bpy.context.scene.use_nodes = False

                    if enable_roughness:
                        for i in range(len(bsdf_nodes)):
                            links = materials[i].node_tree.links
                            roughness_input_link = bsdf_nodes[i].inputs['Roughness'].links[0] if bsdf_nodes[i].inputs['Roughness'].links else None
                            roughness_output = roughness_input_link.from_socket
                            links.new(roughness_output, output_nodes[i].inputs['Surface'])

                        bpy.context.scene.render.filepath = os.path.join(roughness_path,f"sample_{frame_num}.png")
                        bpy.ops.render.render(write_still=True)
                    
                    if enable_metallic:
                        for i in range(len(bsdf_nodes)):
                            links = materials[i].node_tree.links
                            metallic_input_link = bsdf_nodes[i].inputs['Metallic'].links[0] if bsdf_nodes[i].inputs['Metallic'].links else None
                            metallic_output = metallic_input_link.from_socket
                            links.new(metallic_output, output_nodes[i].inputs['Surface'])

                        bpy.context.scene.render.filepath = os.path.join(metallic_path,f"sample_{frame_num}.png")
                        bpy.ops.render.render(write_still=True)
                    
                    if enable_albedo:
                        bpy.context.scene.render.image_settings.file_format = 'PNG'
                        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                        for i in range(len(bsdf_nodes)):
                            links = materials[i].node_tree.links
                            albedo_input_link = bsdf_nodes[i].inputs['Base Color'].links[0] if bsdf_nodes[i].inputs['Base Color'].links else None
                            albedo_output = albedo_input_link.from_socket
                            links.new(albedo_output, output_nodes[i].inputs['Surface'])

                        bpy.context.scene.render.filepath = os.path.join(albedo_path,f"sample_{frame_num}.png")
                        bpy.ops.render.render(write_still=True)

                    if enable_post and enable_final:
                        sam_process_rawdata.post_process_final(os.path.join(final_path,f"sample_{frame_num}.png"))

                    if enable_post and enable_mask:
                        sam_process_rawdata.post_process_mask(os.path.join(final_path,f"sample_{frame_num}.png"))

                    if enable_post and enable_depth:
                        sam_process_rawdata.post_process_depth(os.path.join(depth_path,f"sample_{frame_num}.exr"))
                    
                    if enable_post and enable_normal:
                        sam_process_rawdata.post_process_normal(os.path.join(normal_path,f"sample_{frame_num}.exr"), camera.matrix_world.to_3x3().to_4x4().inverted())#去掉包含的scale和位移矩阵

                    if enable_post and enable_poses:
                        cam_poses.append(np.array(camera.matrix_world))
                    frame_num = frame_num + 1

            if enable_post and enable_poses:
                cam_poses_np = np.array(cam_poses)
                np.save(os.path.join(root_path,"poses.npy"), cam_poses_np)

            if enable_faceid:
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
                bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                bpy.context.scene.render.engine = 'BLENDER_EEVEE'
                bpy.context.scene.render.filter_size = 0
                create_face_index_material()
                
                for k in range(infos_len):
                    frame_num = 0
                    handle_data = scene_infos[k]
                    config.handle_scene_state(handle_data)
                    cam_angles = config.cam_angles(handle_data['elevation'])
                    
                    for rot_value in cam_angles:
                        bpy.context.scene.frame_set(frame_num)
                        radians_value = [math.radians(deg) for deg in rot_value]
                        radians_value[0] = -radians_value[0]
                        model_point.rotation_euler = radians_value
                        bpy.context.scene.render.filepath = os.path.join(faceid_path,f"faceid_{frame_num}.exr")
                        bpy.ops.render.render(write_still=True)
                        sam_process_rawdata.post_process_faceid(bpy.context.scene.render.filepath)
                        frame_num = frame_num + 1

            select_all_childs(render_obj)
            bpy.ops.object.delete()

            for mesh in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)

            for material in bpy.data.materials:
                bpy.data.materials.remove(material)

            for image in bpy.data.images:
                bpy.data.images.remove(image)