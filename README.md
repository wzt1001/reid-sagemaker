## Brief Introduction
The repo is 

## Deployment

Copy `multiple_datasets_trained_model.pth` and `resnet50_ibn_a-d9d0bb7b.pth` to `./weights/` folder. Please ask author for weight files.

Copy `config.sample.yaml` to `config.yaml` and check parameters.

```sh
pip install opencv-python -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install -r requirement.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## Start
Start an mock stream demo and see output video
```sh
python main.py --use-mock-stream --save-vid
```
Start the real-time service in microservice mode
```sh
python main.py
```
For sending real-time tracks to endpoint, please refer to sample.npy for an example. Below is a simple example in json form. Please send the request in x-www-urlcoded POST request.  
```json
[
   {
      "track_id": "6a9bfa9e-bcc8-4d0b-8f9d-9ce151a440c2", # track 名称
      "max_score": "0.71549374", # 轨迹中最大的置信度, optional
      "max_size": "14933.0", # 轨迹中最大图像分辨率 (长x宽), optional
      "best_image": "track-test/2023-03-21-23/6a9bfa9e-bcc8-4d0b-8f9d-9ce151a440c2-best.jpg", # 最大分辨率图像地址
      "large_image": "track-test/2023-03-21-23/6a9bfa9e-bcc8-4d0b-8f9d-9ce151a440c2-large.jpg",
      "frame_idx": 48, # 出现帧ID
      "start_time": "2023-03-21 23:33:42", # 出现时间
      "end_time": "2023-03-21 23:33:57", # 结束时间
      "camera_id": "camera_a", # camera id
      "trajectory": [
         [
            559.5,
            707.5,
            109,
            137
         ], # 轨迹的bounding box，以top-left-width-height的结构
         ...
      ]
   },
   ...
]
```

## TO-DOs
1. visualize people order should be backwards, and set a limit for total people displayed
