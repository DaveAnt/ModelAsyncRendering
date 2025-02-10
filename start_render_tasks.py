import os
import json
import time
import config
import subprocess

def fetch_idle_tasks():
    for i in range(len(render_tasks)):
        if (render_tasks[i] == None) or (render_tasks[i].poll() != None):
            return i
    return -1

def get_model_files(path,model_files):
    for file_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_name)):
            get_model_files(os.path.join(path, file_name), model_files)
        elif file_name.endswith('.gltf') or file_name.endswith('.glb') or file_name.endswith('.fbx') or file_name.endswith('.obj'):
            model_files.append(os.path.join(path[path_len:], file_name))
    return model_files


if __name__ == '__main__':
    path_len = len(config.input_path)
    model_files = get_model_files(config.input_path,[])
    with open('renders.json', 'w') as file:
        json.dump(model_files, file)

    index = 0
    model_count = len(model_files)
    render_tasks = [None] * config.task_num
    batch_num = min(int(model_count / config.task_num),config.batch_num)
    while index < model_count:
        task_id = fetch_idle_tasks()
        if task_id != -1:
            print(str(index),str(batch_num))
            command = f"python {os.path.join(config.prj_path,'sample_render_utils.py')} -- {index} {batch_num}"
            render_tasks[task_id] = subprocess.Popen(command, shell=True)
            index = index + batch_num
        else:
            time.sleep(1)
    print("finish!")