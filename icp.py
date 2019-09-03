from __future__ import print_function
import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages) 
import numpy as np
import cv2

if __name__ == '__main__':
    # load point clouds
    model = cv2.ppf_match_3d.loadPLYSimple('/home/xia/bobbins/model01.ply')
    scene = cv2.ppf_match_3d.loadPLYSimple('/home/xia/bobbins/model02.ply')
    # compute normals
    model_norm = cv2.ppf_match_3d.computeNormalsPC3d(model, 5, False, (0,0,0))
    model_norm = model_norm[1]   
    scene_norm = cv2.ppf_match_3d.computeNormalsPC3d(scene, 5, False, (0,0,0))
    scene_norm = scene_norm[1]
    # Super4PCS transform
    pose_super = np.array([[-0.325427, 0.875482, 0.35725, 0.28899],
        [-0.940145, -0.339983, -0.0232308, 0.43661],
        [0.101121, -0.343427, 0.93372, 0.0744636],
        [0, 0, 0, 1]])
    model_super = cv2.ppf_match_3d.transformPCPose(model_norm, pose_super)
    # icp
    icp = cv2.ppf_match_3d_ICP(100)
    pose_icp = icp.registerModelToScene(model_super, scene_norm)
    # icp transform
    model_icp = cv2.ppf_match_3d.transformPCPose(model_super, pose_icp[2])
    # save pointcloud
    #cv2.ppf_match_3d.writePLY(model_icp, 'model01_icp.ply')
    