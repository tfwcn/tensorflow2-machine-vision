import numpy as np

def Get_TPFP(data, class_id, thresh=0.5):
    '''
    计算TPFP
    data:[{
        'image_path': '*.jpg',
        'groud_truth': [[x1,y1,x2,y2,class_id], ...],
        'prediction': [[x1,y1,x2,y2,class_id,score], ...],
        }]

    return
    tp:[[tp,score], ...]
    '''
    tp = []
    groud_truth_num = 0
    for d in data:
        # (1, groud_truth_num, 6)
        groud_truth = np.array(d['groud_truth'], dtype=np.float)
        groud_truth = groud_truth[groud_truth[..., 4]==class_id]
        groud_truth = np.expand_dims(groud_truth, axis=0)
        groud_truth_num += groud_truth.shape[1]
        # print('groud_truth:', groud_truth.shape, groud_truth)
        # (prediction_num, 1, 6)
        prediction = np.array(d['prediction'], dtype=np.float)
        prediction = prediction[prediction[..., 4]==class_id]
        prediction = np.expand_dims(prediction, axis=1)
        # print('prediction:', prediction.shape, prediction)
        if groud_truth.shape[1] == 0 or prediction.shape[0] == 0:
            continue
        # 计算IOU
        groud_truth_mins = groud_truth[..., 0:2]
        groud_truth_maxes = groud_truth[..., 2:4]
        groud_truth_wh = groud_truth_maxes - groud_truth_mins
        prediction_mins = prediction[..., 0:2]
        prediction_maxes = prediction[..., 2:4]
        prediction_wh = prediction_maxes - prediction_mins
        intersect_mins = np.maximum(groud_truth_mins, prediction_mins)
        intersect_maxes = np.minimum(groud_truth_maxes, prediction_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        groud_truth_area = groud_truth_wh[..., 0] * groud_truth_wh[..., 1]
        prediction_area = prediction_wh[..., 0] * prediction_wh[..., 1]
        # (prediction_num, groud_truth_num)
        iou = intersect_area / (groud_truth_area + prediction_area - intersect_area)
        # print('iou:', iou.shape, iou)
        # print('iou:', iou>=thresh)
        # print('iou:', np.argmax(iou, axis=0))
        # print('iou:', np.argmax(iou, axis=1))
        tp_one = np.zeros((prediction.shape[0],))
        # 按groud_truth查找最大iou的prediction下标
        iou_max = np.argmax(iou, axis=0)
        for i in range(iou_max.shape[0]):
            if iou[iou_max[i],i]>=thresh:
                tp_one[iou_max[i]] = 1
        tp_one = np.expand_dims(tp_one, axis=-1)
        tp_one = np.concatenate([tp_one, prediction[:, 0,5:6]], axis=-1)
        # print('tp_one:', tp_one)
        tp.append(tp_one)
    tp = np.array(tp)
    tp = tp.reshape((-1, 2))
    # 排序
    tp = tp[np.argsort(tp[:, 1])[::-1], :]
    # print('tp:', tp.shape)
    # print('tp:', tp)
    # print('groud_truth_num:', groud_truth_num)
    return tp, groud_truth_num

def Get_AP(data, class_id, thresh=0.5):
    '''计算AP'''
    # 计算TP
    tp, groud_truth_num = Get_TPFP(data, class_id=class_id, thresh=thresh)
    # 计算precision和recall
    precision_list = []
    recall_list = []
    tp_sum = 0.0
    for i in range(tp.shape[0]):
        if tp[i][0] == 1:
            tp_sum += 1.0
        precision = tp_sum / (i+1)
        precision_list.append(precision)
        recall = tp_sum / groud_truth_num
        recall_list.append(recall)
    # print('tp_sum:', tp_sum)
    # print('precision_list:', precision_list)
    # print('recall_list:', recall_list)
    # 计算AP
    mrec = np.concatenate(([0.], precision_list, [1.]))
    mpre = np.concatenate(([0.], recall_list, [0.]))
    # print(mpre)
    # compute the precision envelope
    # 使precision一直下降，去掉波动
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
 
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # print('ap:', ap)
    return ap

def Get_mAP(data, class_num, thresh=0.5):
    '''计算mAP'''
    ap_sum = 0.0
    for class_id in range(class_num):
        # print('class_id:', class_id)
        ap = Get_AP(data, class_id, thresh=thresh)
        ap_sum += ap
    return ap_sum / class_num



def Get_mAP_one(groud_truth, prediction, class_num, thresh=0.5):
    '''计算mAP'''

    data = [
        {
            'image_path': '*.jpg',
            'groud_truth': groud_truth,
            'prediction': prediction,
        }
    ]
    # print('data:', data)
    return Get_mAP(data, class_num=class_num, thresh=thresh)



def main():
    data = [
        {
            'image_path': '*.jpg',
            'groud_truth': [[1,1,2,2,1], [1,1,2,2,2], [1,1.3,2.4,2,1], [3,1,4,2,2]],
            'prediction': [[1.1,1,2.1,2.2,1,0.8], [1.2,1.2,2.2,2.2,2,0.7], [1.1,1.3,2.4,2.1,1,0.6], [1.1,1.1,2.1,2.1,1,0.9]],
        },
        {
            'image_path': '*.jpg',
            'groud_truth': [[1,1,2,2,1], [1,1,2,2,2], [1,1.3,2.4,2,1], [3,1,4,2,2], [3,1,4,2,0]],
            'prediction': [[1.1,1,2.1,2.2,1,0.8], [1.2,1.2,2.2,2.2,2,0.7], [1.1,1.3,2.4,2.1,1,0.7], [1.1,1.1,2.1,2.1,1,0.6]],
        }
    ]
    mAP = Get_mAP(data, class_num=3, thresh=0.5)
    print('mAP:', mAP)


if __name__ == '__main__':
    main()