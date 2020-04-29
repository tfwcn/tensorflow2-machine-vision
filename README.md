# tensorflow2-yolov4

基于tensorflow2的yolov4实现

### 实现内容

CSPDarknet53 + SPP + PAN (已实现)
CIoU-loss (已实现)
Mish激活函数 (已实现)
Crossstage partial connections(CSP) (已实现)
CutMix (未实现)
Mosaic data augmentation (未实现)
DropBlock regularization (未实现)
Class label smoothing (未实现)
Multiinput weighted residual connections(MiWRC) (未实现)
CmBN (未实现)
Self Adversarial Training (未实现)
Eliminate gridsensitivity (未实现)
Using multiple anchors for a single groundtruth (未实现)
Cosine annealing scheduler (未实现)
Optimal hyper-parameters (未实现)
Random training shapes (未实现)
Mish activation (未实现)
SAM-block (未实现)
DIoU-NMS (未实现)

### 素材标注

素材标注使用Labelme
https://github.com/tfwcn/labelme

### 训练

```bash
python ./ai_api/yolo_v4/train_label.py --file_path "素材目录" --batch_size 8
```

### 测试

启动服务：
```bash
python manage.py runserver 0.0.0.0:8080
```

浏览器打开：http://127.0.0.1:8080/static/object_detection/predict_image_read.html