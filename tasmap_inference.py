import os
from tqdm import tqdm
import time
import argparse
import json
from dataset.tasmap import TASMapDataset
from dataset.scannet import ScanNetDataset


def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, default='scene0000_00')
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='tasmap')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset == 'scannet':
        dataset = ScanNetDataset(args.seq_name)
    elif args.dataset == 'tasmap':
        dataset = TASMapDataset(args.seq_name)
    else:
        print(args.dataset)
        raise NotImplementedError
    return dataset


# def execute_commands(commands_list, command_type, process_num):
#     print('====> Start', command_type)
#     from multiprocessing import Pool
#     pool = Pool(process_num)
#     for _ in tqdm(pool.imap_unordered(os.system, commands_list), total=len(commands_list)):
#         pass
#     pool.close()
#     pool.join()
#     pool.terminate()
#     print('====> Finish', command_type)

def execute_commands(commands_list, command_type, process_num):
    print('====> Start', command_type)
    from multiprocessing import Pool
    pool = Pool(process_num)
    for _ in pool.imap_unordered(os.system, commands_list):
        pass
    pool.close()
    pool.join()
    pool.terminate()
    print('====> Finish', command_type)

def get_seq_name_list(dataset):
    if dataset == 'scannet':
        file_path = 'splits/scannet_test.txt'
    elif dataset == 'tasmap':
        file_path = 'splits/tasmap.txt'
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    return seq_name_list

def parallel_compute(general_command, command_name, resource_type, cuda_list, seq_name_list):
    cuda_num = len(cuda_list)
    
    if resource_type == 'cuda':
        commands = []
        for i, cuda_id in enumerate(cuda_list):
            process_seq_name = seq_name_list[i::cuda_num]
            if len(process_seq_name) == 0:
                continue
            process_seq_name = '+'.join(process_seq_name)
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command % process_seq_name}'
            commands.append(command)
        execute_commands(commands, command_name, cuda_num)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)

def get_label_text_feature(cuda_id):
    label_text_feature_path = 'data/text_features/matterport3d.npy'
    if os.path.exists(label_text_feature_path):
        return
    command = f'CUDA_VISIBLE_DEVICES={cuda_id} python -m semantics.extract_label_featrues'
    os.system(command)

def main(args):
    dataset = args.dataset
    config = args.config
    cropformer_path = args.cropformer_path

    if dataset == 'scannet':
        root = 'data/scannet/processed'
        image_path_pattern = 'color/*0.jpg' # stride = 10
        gt = 'data/scannet/gt'
    elif dataset == 'tasmap':
        root = 'data/tasmap/processed'
        image_path_pattern = 'color/*.jpg' # stride = 1
        gt = 'data/tasmap/gt'


    t0 = time.time()
    seq_name_list = get_seq_name_list(dataset)
    print('There are %d scenes' % len(seq_name_list))
    
    # # Step 1: use Cropformer to get 2D instance masks for all sequences.
    # parallel_compute(f'python third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --seq_name_list %s --opts MODEL.WEIGHTS {cropformer_path}', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)
    parallel_compute(f'python third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file /workspace/MaskClustering/third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --seq_name_list %s --opts MODEL.WEIGHTS {cropformer_path}', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)

    # # Step 2: Mask clustering using our proposed method.
    parallel_compute(f'python main.py --config {config} --seq_name_list %s', 'mask clustering', 'cuda', CUDA_LIST, seq_name_list)
    
    print('total time', (time.time() - t0)//60, 'min')
    print('Average time', (time.time() - t0) / len(seq_name_list), 'sec')

    # Visualize Mask
    parallel_compute(f'python -m visualize.vis_mask --config {config} --seq_name %s', 'Visualize Mask', 'cuda', CUDA_LIST, seq_name_list)

    # Visualize Scene
    parallel_compute(f'python -m visualize.vis_scene --config {config} --seq_name %s', 'Visualize Scene', 'cuda', CUDA_LIST, seq_name_list)

    print("====> To Visualize in PyViz3D:")
    for seq_name in seq_name_list:
        print(f"python -m http.server 6010 --directory /workspace/MaskClustering/data/vis/{seq_name}")


if __name__ == '__main__':
    CUDA_LIST = [0]
    args = get_args()
    main(args)