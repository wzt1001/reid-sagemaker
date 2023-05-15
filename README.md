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
Start the service in microservice mode
```sh
python main.py
```

## TO-DOs
1. visualize people order should be backwards, and set a limit for total people displayed