import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from plyfile import PlyData

def rotate(file_name, mesh_file,meta_file,new_dir):
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        mesh_vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        mesh_vertices[:,0] = plydata['vertex'].data['x']
        mesh_vertices[:,1] = plydata['vertex'].data['y']
        mesh_vertices[:,2] = plydata['vertex'].data['z']

    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    
    plydata['vertex'].data['x'] = pts[:,0]
    plydata['vertex'].data['y'] = pts[:,1]
    plydata['vertex'].data['z'] = pts[:,2]
    
    plydata.write(os.path.join(new_dir,file_name))
    

if __name__ == '__main__':
    base_dir = '/workspace/3DETR/scans'
    new_dir = '/workspace/3DETR/scans/rotated_vh_clean_2_files'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    file_val = "/workspace/3DETR/meta_data/scannetv2_val.txt"
    lines = open(file_val).readlines()
    
    for line in lines:
        line = line.strip('\n')
        # print(line)
        file_name = line + '_vh_clean_2.ply'
        meta_file = line + '.txt'
        file_path = os.path.join(base_dir,"vh_clean_2_files",file_name)
        meta_file = os.path.join(base_dir,"metadata", line,meta_file)
        rotate(file_name, file_path, meta_file, new_dir)