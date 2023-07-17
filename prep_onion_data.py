import numpy as np
from pypcd import pypcd
import os
import json
import logging
import glob
import random
import pickle
import collections


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class bbox3d:
    def __init__(self, x=0, y=0, z=0, dx=0, dy=0, dz=0, yaw=0, fname='', label='') -> None:
        # centroids
        self.x = x
        self.y = y
        self.z = z
        # dimensions
        self.dx = dx
        self.dy = dy
        self.dz = dz
        # rotation in Z axis
        self.yaw = yaw
        self.label = label
        self.fname = fname
        
    def __str__(self) -> str:
        s1 = f'fname: {self.fname}\n'
        s2 = f'label: {self.label}\n'
        s3 = f'x: {self.x}, y: {self.y}, z: {self.z}\n'
        s4 = f'dx: {self.dx}, dy: {self.dy}, dz: {self.dz}\n'
        s5 = f'yaw: {self.yaw}\n'
        return s1+s2+s3+s4+s5
        
    def load_dt(self, dt:dict):
        self.x, self.y, self.z = dt['centroid']['x'], dt['centroid']['y'], dt['centroid']['z']
        self.dx, self.dy, self.dz = dt['dimensions']['length'], dt['dimensions']['width'], dt['dimensions']['height']
        self.yaw = dt['rotations']['z']
        self.label = dt['name']
        self.z -= self.dz/2 # bottom-view coordinates
        
    def tolist(self):
        return [self.x, self.y, self.z, self.dx, self.dy, self.dz, self.yaw]
          

def load_pcd(path_pcd):
    pcd_data = pypcd.PointCloud.from_path(path_pcd)
    points = np.zeros([pcd_data.width, 4], dtype=np.float32)
    points[:, 0] = pcd_data.pc_data['x'].copy()
    points[:, 1] = pcd_data.pc_data['y'].copy()
    points[:, 2] = pcd_data.pc_data['z'].copy()
    if len(pcd_data.fields)>3:
        rgb_key = pcd_data.fields[-1]
        points[:, 3] = pcd_data.pc_data[rgb_key].copy().astype(np.float32)
    else:
        points = points[:,:3]
    return points


def get_pcd_range(points):
    X,Y,Z = points[:,0],points[:,1],points[:,2]
    return X.min(),Y.min(),Z.min(),X.max(),Y.max(),Z.max()


def write_to_bin(points, path_out):
    _,ext = os.path.splitext(path_out)
    if ext!='.bin':
        raise Exception('the pointclouds must be saved as *.bin files')
    with open(path_out, 'wb') as f:
        f.write(points.tobytes())

        
def load_label(path_json):
    bboxes = []
    with open(path_json,'r') as f:
        dt = json.load(f)
        # path = dt['path']
        fname = dt['filename']
        for ob in dt['objects']:
            box3d = bbox3d()
            box3d.load_dt(ob)
            box3d.fname = fname
            logger.debug(box3d)
            bboxes.append(box3d)
    return bboxes


def write_to_txt(bboxes, path_out):
    _,ext = os.path.splitext(path_out)
    if ext!='.txt':
        raise Exception('the labels must be saved as *.txt files')
    with open(path_out,'w') as f:
        for b in bboxes:
            f.write(f'{b.x} {b.y} {b.z} {b.dx} {b.dy} {b.dz} {b.yaw} {b.label}\n')
            

def map_to_dict(sample_idx:int, path_bin:str, bboxes:list, class_to_id:dict, num_feats:int=4):
    dt = {}
    dt['sample_idx'] = sample_idx
    dt['lidar_points'] = {'lidar_path': f'{path_bin}','num_pts_feats':num_feats}
    dt['instances'] = []
    # dummy data
    dt['images'] = {'R0_rect':np.eye(4),'CAM2':{}}
    dt['images']['CAM2']['height'] = 0
    dt['images']['CAM2']['width'] = 0
    dt['images']['CAM2']['cam2img'] = np.eye(4)
    dt['images']['CAM2']['lidar2cam'] = np.eye(4)
    dt['images']['CAM2']['lidar2img'] = np.eye(4)
    
    for bbox in bboxes:
        tmp = {
            'bbox_3d':bbox.tolist(),
            'bbox_label_3d':class_to_id[bbox.label],
            # dummy info
            'bbox':[0]*4,
            'bbox_label':class_to_id[bbox.label],
            'truncated':0,
            'occluded':0,
            'alpha':0,
            'score':0,
        }
        dt['instances'].append(tmp)
    return dt
        
        
            
if __name__ == '__main__':
    path_jsons = 'raw_data/labels_bp'
    path_pcds = 'raw_data/2023-06-20'
    val_percentage = 0.1
    path_out = 'data/onion'
    categories = 'root,stem' # must match with METAINFO in onion_dataset.py
    
    metainfo = {'categories':{}}
    idx = 0
    classes = categories.split(',')
    for cat in classes:
        metainfo['categories'].update({
            cat:idx
        })
        idx += 1
    
    paths = glob.glob(os.path.join(path_jsons, '*.json'))
    random.seed(777)
    random.shuffle(paths)
    train_len = int(len(paths)*(1-val_percentage))
    val_len = len(paths)-train_len
    if not train_len or not val_len:
        raise Exception('after train-val split, the number of training or validation samples is zero')
    valset = set(paths[train_len:])
    
    list_train = []
    list_val = []
    annots_train = {
        'metainfo':metainfo,
        'data_list':[],
    }
    annots_val = {
        'metainfo':metainfo,
        'data_list':[],
    }
    stats = {c:collections.defaultdict(list) for c in classes}
    final_range = np.array([np.inf,np.inf,np.inf,-np.inf,-np.inf,-np.inf])
    for i,p in enumerate(paths):
        fname = os.path.basename(p)
        fname = os.path.splitext(fname)[0]
        
        # load labels
        bboxes = load_label(p)
        
        # get stats of labels
        for box in bboxes:
            stats[box.label]['z'] += [box.z]
            stats[box.label]['dx'] += [box.dx]
            stats[box.label]['dy'] += [box.dy]
            stats[box.label]['dz'] += [box.dz]
        
        # write labels
        tmp_out = os.path.join(path_out,'labels')
        os.makedirs(tmp_out, exist_ok=1)
        write_to_txt(bboxes, os.path.join(tmp_out,fname+'.txt'))
        
        if not len(bboxes):
            logging.warning(f'cannot find any 3D boxes in {p}. Skip')
            continue
        
        # load pointcloud
        points = load_pcd(os.path.join(path_pcds, fname+'.pcd'))
        # get pcd range in the format of [xmin,ymin,zmin,xmax,ymax,zmax]
        range = get_pcd_range(points)
        final_range[:3] = np.min([final_range[:3],range[:3]],axis=0)
        final_range[3:] = np.max([final_range[3:],range[3:]],axis=0)
        logger.debug(f'current range: {range}')
        logger.debug(f'final range: {final_range}')
        tmp_out = os.path.join(path_out,'points')
        os.makedirs(tmp_out, exist_ok=1)
        write_to_bin(points, os.path.join(tmp_out,fname+'.bin'))
        
        # create annotation dicts
        annot = map_to_dict(i,fname+'.bin',bboxes,metainfo['categories'],points.shape[1])
        
        # assign fname to train/val sets
        if p in valset:
            list_val.append(fname+'.bin')
            annots_val['data_list'].append(annot)
        else:
            list_train.append(fname+'.bin')
            annots_train['data_list'].append(annot)
    
    # logs        
    logger.info(metainfo['categories'])
    for c in classes:
        logger.info(f"class {c}'s avg z: {np.mean(stats[c]['z'])}")
        logger.info(f"class {c}'s avg dx,dy,dz: {[np.mean(stats[c]['dx']),np.mean(stats[c]['dy']),np.mean(stats[c]['dz'])]}")
    logger.info(f'actual final range: {final_range}')
    final_range = np.concatenate([np.floor(final_range[:3]),np.ceil(final_range[3:])],axis=0)
    logger.info(f'final range: {final_range}')
    
    # write out
    tmp_out = os.path.join(path_out, 'ImageSets')
    os.makedirs(tmp_out, exist_ok=1)
    with open(os.path.join(tmp_out,'train.txt'),'w') as f:
        for fname in list_train:
            f.write(f'{fname}\n')
    with open(os.path.join(tmp_out,'val.txt'),'w') as f:
        for fname in list_val:
            f.write(f'{fname}\n')
            
    with open(os.path.join(path_out,'annotation_train.pkl'),'wb') as f:
        pickle.dump(annots_train, f)
    
    with open(os.path.join(path_out,'annotation_val.pkl'),'wb') as f:
        pickle.dump(annots_val, f)
        