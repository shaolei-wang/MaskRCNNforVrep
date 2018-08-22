image_id 2
/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py:1272: VisibleDeprecationWarning: boolean index did not match indexed array along dimension 0; dimension is 2 but corresponding boolean dimension is 128
  class_ids = class_ids[_idx]
ERROR:root:Error processing image {'id': 6, 'source': 'shapes', 'path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/rgb/55.jpg', 'width': 640, 'height': 480, 'mask_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/mask/55.png', 'yaml_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/labelme_json/55_json/info.yaml'}
Traceback (most recent call last):
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1717, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1272, in load_image_gt
    class_ids = class_ids[_idx]
IndexError: index 127 is out of bounds for axis 1 with size 2
image_id 5
/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py:1272: VisibleDeprecationWarning: boolean index did not match indexed array along dimension 0; dimension is 2 but corresponding boolean dimension is 128
  class_ids = class_ids[_idx]
ERROR:root:Error processing image {'id': 6, 'source': 'shapes', 'path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/rgb/55.jpg', 'width': 640, 'height': 480, 'mask_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/mask/55.png', 'yaml_path': '/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/labelme_json/55_json/info.yaml'}
Traceback (most recent call last):
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1717, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/home/enwhsaa/ros/train_model/Mask_RCNN-master/mrcnn/model.py", line 1272, in load_image_gt
    class_ids = class_ids[_idx]
IndexError: index 127 is out of bounds for axis 1 with size 2

