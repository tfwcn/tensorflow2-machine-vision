## 转换权重

yolo权重转h5，yolo权重请在官网下载
```
python ./ai_api/ai_models/yolo_v3/convert.py ./data/yolo_v3_weights/yolov3.cfg ./data/yolo_v3_weights/yolov3.weights ./data/yolo_v3_weights/yolo.h5 --plot_model --weights_only
```

h5转tf，如果类别数有变化，需重新转换
```
python ./ai_api/ai_models/yolo_v3/convert_tf2.py --old_weights_path ./data/yolo_v3_weights/yolo.h5 --output_path ./data/yolo_v3_weights/tf2_weights/tf2_weights.ckpt --classes_num 80
```

## 训练

```
python ./ai_api/ai_models/yolo_v3/train.py --trainData "./coco2017/train2017" --valData "./coco2017/val2017" --trainLabels "./data/coco_train_labels.txt" --valLabels "./data/coco_test_labels.txt" --classesFile "./data/coco_classes.txt" --anchorsFile "./data/coco_anchors.txt" --batchSize 4
```

## 测试

```
python ./ai_api/ai_models/yolo_v3/test.py --imageFile "./data/img.png" --anchorsFile "./data/coco_anchors.txt" --classesFile "./data/coco_classes.txt" --modelPath "./data/yolo_v3_weights/tf2_weights/"
```
