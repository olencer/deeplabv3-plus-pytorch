import os
import cv2
import utils.data as data

# import json

if __name__ == "__main__":
    original_folder = os.path.join('datasets', 'original')
    dataset_folder = os.path.join('datasets', 'PlaneEngine')

    generator_label_folder = os.path.join('datasets', 'before')
    
    rgb_folder = os.path.join(dataset_folder, 'Images')
    depth_folder = os.path.join(dataset_folder, 'Depths')
    # intrinsics_folder = os.path.join(dataset_folder, 'Intrinsics')

    if not os.path.exists(original_folder):
        os.mkdir(original_folder)
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(generator_label_folder):
        os.mkdir(generator_label_folder)
    if not os.path.exists(rgb_folder):
        os.mkdir(rgb_folder)
    if not os.path.exists(depth_folder):
        os.mkdir(depth_folder)
    # if not os.path.exists(intrinsics_folder):
    #     os.mkdir(intrinsics_folder)   
    
    for i in range(1, 113):
        base_name = str(i).zfill(3)
        
        data_name       = base_name + '.zdf'
        img_name        = base_name + '.jpg'
        # intrinsics_name = base_name + '.json'

        input = data.Data(os.path.join(original_folder, data_name))

        rgb = input.get_rgb()
        cv2.imwrite(os.path.join(generator_label_folder, img_name), rgb)

        depth = input.get_depth()
        cv2.imwrite(os.path.join(depth_folder, img_name), depth)

        # intrinsics = input.get_intrinsics()
        # with open(os.path.join(base_folder, intrinsics_name), 'w') as file:
        #     json.dump(intrinsics, file, indent=4)
        # file.close()

        print("saved " + base_name + '.')
