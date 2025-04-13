import numpy as np

import zivid
import zivid.experimental.calibration

class Data:

    def __init__(self, data_path):
        self.zividApp = zivid.Application()
        self.frame = zivid.Frame(data_path)
        self.point_cloud = self.frame.point_cloud()

    def get_rgb(self):
        rgba = self.point_cloud.copy_data("rgba")
        return rgba[:, :, :-1]


    def get_depth(self):
        depth_float = self.point_cloud.copy_data("z")
        depth_float[np.isnan(depth_float)] = 0
        depth = ((depth_float - np.nanmin(depth_float)) / (np.nanmax(depth_float) - np.nanmin(depth_float)) * 255).astype(np.uint8)

        return depth
    
    
    def get_intrinsics(self):
        intrinsics = zivid.experimental.calibration.estimate_intrinsics(self.frame).camera_matrix
        intrinsics_json = { 'fx': intrinsics.fx, 'fy': intrinsics.fy,
                            'cx': intrinsics.cx, 'cy': intrinsics.cy  }
        return intrinsics_json
