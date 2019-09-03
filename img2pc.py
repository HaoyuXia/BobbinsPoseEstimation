import numpy as np
import os

def setCamera(px=325.26110, py=242.04899, fx=572.441140, fy=573.57043):
    camera = {'px':px, 'py':py, 'fx':fx, 'fy':fy}
    return camera

def createPointCloud(color, depth, rgb:bool, name:str = None):
    #color = cv2.imread(color_name, cv2.IMREAD_COLOR)
    #color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
    camera = setCamera()
    pointcloud = []
    depth = np.array(depth, dtype='float')
    if rgb == True:        
        for row in range(depth.shape[0]):
            for col in range(depth.shape[1]):
                xi = 0
                yi = 0
                zi = depth[row][col] / 1000.0
                if zi != 0.0:
                    xi = zi * (col + 1 - camera['px']) / camera['fx']
                    yi = zi * (row + 1 - camera['py']) / camera['fy']
                    pointcloud.append([xi, yi, zi, 
                              color[row, col, 0], color[row, col, 1], color[row, col, 2]])
    else:
        pointcloud = np.zeros((depth.shape[0]*depth.shape[1], 3), dtype='float')
        for row in range(depth.shape[0]):
            for col in range(depth.shape[1]):
                xi = 0
                yi = 0
                zi = depth[row][col] / 1000.0
                if zi != 0:
                    xi = zi * (col + 1 - camera['px']) / camera['fx']
                    yi = zi * (row + 1 - camera['py']) / camera['fy']
                    pointcloud.append([xi, yi, zi])
    if name != None:
        writePLY(pointcloud, name)
    return pointcloud

def writePLY(pointcloud, ply_name):
    point_num = len(pointcloud)
    vert_num = len(pointcloud[0])
    with open(ply_name, 'w') as ply_file:
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('element vertex %d\n'%(point_num))
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        if vert_num == 6:
            ply_file.write('property uchar red\n')
            ply_file.write('property uchar green\n')
            ply_file.write('property uchar blue\n')
        ply_file.write('end_header\n')        
        for pi in range(point_num):
            point = pointcloud[pi]
            ply_file.write('%f %f %f '%(point[0], point[1], point[2]))
            if vert_num == 6:
                ply_file.write('%d %d %d'%(point[3], point[4], point[5]))
            ply_file.write('\n')        
    print('ply saved to %s/%s'%(os.getcwd(), ply_name))

