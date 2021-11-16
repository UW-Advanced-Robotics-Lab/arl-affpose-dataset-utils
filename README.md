### This repo contains tools used to prep the ARL AffPose Dataset.

![Alt text](samples/dataset.png?raw=true "Title")

The ARL AffPose Dataset can be found here - [Real Images](https://drive.google.com/drive/folders/1gP-vQVuDKdhCxdViRxAeoH8v99sTfPXi?usp=sharing) or [Synthetic Images](https://drive.google.com/drive/folders/1X47BIXqyMO9xyoFMCXEGVPP9T0ZDEPUe?usp=sharing).

Object Mesh files can also be found here - [Object Mesh files](https://drive.google.com/file/d/1MwRN3CS2iGLoVmUzkLUrLMryVxJhugWA/view?usp=sharing). 

### ZED Camera.

A Stereolab ZED camera at 1280x720 was used or simulated to capture images. 

To this end, each RGB-D image is annotated with a object 6-DoF Pose using [LabelFusion for Real Images](https://github.com/RobotLocomotion/LabelFusion) or [NDDS for Synthetic Images](https://github.com/NVIDIA/Dataset_Synthesizer).

### LabelFusion or NDDS.

First, configure relative paths for real or synthetic images and object mesh files in `cfg.py`.

Leverage the utils/ folder to view RGB-D images with ground truth annotations. Each annotation should have a 1.) bounding box, 2,) segmentation mask and 3.) projected point cloud.

`format_labelfusion_obj_to_obj_part_pose.py` transforms the object 6-DoF pose to the object part 6-DoF pose.

### AffPose.

Scripts to view the prepared dataset with RGB-D images and ground truth annotations with affordance labels and object part 6-DoF pose.

 
