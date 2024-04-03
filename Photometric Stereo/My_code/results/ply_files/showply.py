import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
filepath =  'venus.ply'
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])
    
show_ply(filepath)
