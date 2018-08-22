ERROR:root:Error processing image {'id': 33, 'source': 'shapes', 'path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/rgb/rgb_6.jpg', 'width': 640, 'height': 480, 'mask_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/mask/rgb_6.png', 'yaml_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/labelme_json/rgb_6_json/info.yaml'}
Traceback (most recent call last):
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1717, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1272, in load_image_gt
    class_ids = class_ids[_idx]
IndexError: boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 128
image_id 8
