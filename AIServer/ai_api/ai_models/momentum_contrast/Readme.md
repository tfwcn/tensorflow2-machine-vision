
## 训练

```
python ./ai_api/ai_models/momentum_contrast/train.py --trainData "./coco2017/train2017" --valData "./coco2017/val2017" --batchSize 4
```

## 保存MoCo训练完的权重

```
python ./ai_api/ai_models/momentum_contrast/save_model.py --classes_num 80
```

## 训练目标检测模型

```
python ./ai_api/ai_models/momentum_contrast/train_object_detection.py --trainData "./coco2017/train2017" --valData "./coco2017/val2017" --trainLabels "./data/coco_train_labels.txt" --valLabels "./data/coco_test_labels.txt" --classesFile "./data/coco_classes.txt" --anchorsFile "./data/coco_anchors.txt" --batchSize 4
```

## 测试

```
python ./ai_api/ai_models/momentum_contrast/test_object_detection.py --imageFile "./data/img.png" --anchorsFile "./data/coco_anchors.txt" --classesFile "./data/coco_classes.txt" --modelPath "./data/momentum_contrast_weights/train_object_detection_weights/"
```