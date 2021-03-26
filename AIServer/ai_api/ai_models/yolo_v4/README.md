# tensorflow2-yolov4

基于tensorflow2的yolov4实现

源码地址：https://github.com/tfwcn/tensorflow2-machine-vision/tree/master/AIServer/ai_api/ai_models/yolo_v4

### 更新日志

#### 2020-07-30
- 通过继承方式，重新封装各层代码
#### 2021-03-15
- 重新封装训练代码

### 实现内容

- CSPDarknet53 + SPP + PAN (已实现)
- CIoU-loss (已实现)
- Mish activation (已实现)
- Crossstage partial connections(CSP) (已实现)
- DIoU-NMS (已实现)
- CutMix (未实现)
- Mosaic data augmentation (未实现)
- DropBlock regularization (效果较差，暂不发布)
- Class label smoothing (未实现)
- Multiinput weighted residual connections(MiWRC) (未实现)
- CmBN (未实现)
- Self Adversarial Training (未实现)
- Eliminate gridsensitivity (未实现)
- Using multiple anchors for a single groundtruth (未实现)
- Cosine annealing scheduler (未实现)
- Optimal hyper-parameters (未实现)
- Random training shapes (未实现)
- SAM-block (未实现)


### 素材标注

素材标注使用Labelme
https://github.com/tfwcn/labelme

### 转换权重

yolo权重转h5，yolo权重请在官网下载：
https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
python ./ai_api/ai_models/yolo_v4/convert.py ./data/yolo_v4_weights/yolov3.cfg ./data/yolo_v4_weights/yolov3.weights ./data/yolo_v4_weights/yolo.h5 --plot_model --weights_only
```

h5转tf，如果类别数有变化，需重新转换
```
python ./ai_api/ai_models/yolo_v4/convert_tf2.py --old_weights_path ./data/yolo_v4_weights/yolo.h5 --output_path ./data/yolo_v4_weights/tf2_weights/tf2_weights.ckpt --classes_num 80
```

### coco2017标签转换

```
python ./ai_api/ai_models/utils/coco.py --dataDir ./coco2017 --dataType train2017 --outDir ./data
```

### 训练

```
python ./ai_api/ai_models/yolo_v4/train.py --trainData "./coco2017/train2017" --valData "./coco2017/val2017" --trainLabels "./data/coco_train_labels.txt" --valLabels "./data/coco_test_labels.txt" --classesFile "./data/coco_classes.txt" --anchorsFile "./data/coco_anchors.txt" --batchSize 4
```

### 测试

```
python ./ai_api/ai_models/yolo_v4/test.py --imageFile "./data/img.png" --anchorsFile "./data/coco_anchors.txt" --classesFile "./data/coco_classes.txt" --modelPath "./data/yolo_v4_weights/tf2_weights/"
```


### 测试

启动服务：
```bash
python manage.py runserver 0.0.0.0:8080
```

浏览器打开：http://127.0.0.1:8080/static/object_detection/predict_image_read.html

### 参考资料：
- https://github.com/AlexeyAB/darknet
- https://github.com/qqwweee/keras-yolo3