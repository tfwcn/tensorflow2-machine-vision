## 训练

### 带平均梯度
```
python ./ai_api/ai_models/facenet/train.py "files_path" --batch_size 45 --learning_rate 0.1 --moving_average --moving_average_decay 0.9999 --loss_decay 0.9 --backbone InceptionResNetV1 --random_crop --lfw_pairs ./lfw/pairs.txt --lfw_dir ./lfw/lfw_mtcnnpy_160/
```

### 普通训练
```
python ./ai_api/ai_models/facenet/train.py ./lfw/lfw_mtcnnpy_182/ --batch_size 45 --learning_rate 0.01  --backbone InceptionResNetV1 --random_crop --lfw_pairs ./lfw/pairs.txt --lfw_dir ./lfw/lfw_mtcnnpy_160/
```

### 测试
```
python ./ai_api/ai_models/facenet/test.py ./lfw/lfw_mtcnnpy_160/Aaron_Eckhart/Aaron_Eckhart_0001.png ./lfw/lfw_mtcnnpy_160/Aaron_Peirsol/Aaron_Peirsol_0001.png ./lfw/lfw_mtcnnpy_160/Aaron_Peirsol/Aaron_Peirsol_0002.png ./lfw/lfw_mtcnnpy_160/Aaron_Peirsol/Aaron_Peirsol_0003.png --backbone InceptionResNetV1
```

### 验证LFW
```
python ./ai_api/ai_models/facenet/validate_on_lfw.py ./lfw/lfw_mtcnnpy_160/ --backbone InceptionResNetV1 --batch_size 45 --lfw_pairs ./lfw/pairs.txt
```