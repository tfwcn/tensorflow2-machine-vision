from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
import skimage.io as io
import os
import pylab
import argparse
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# matplotlib.use('TkAgg')

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', default='annotations_trainval2017')
parser.add_argument('--dataType', default='train2017')
parser.add_argument('--outDir', default='./data')
args = parser.parse_args()

dataDir = args.dataDir
dataType = args.dataType
outDir = args.outDir

annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# 初始化coco api
coco=COCO(annFile)

# 获取coco 类别
catIds = coco.getCatIds()
print('catIds: ', catIds)
cats = coco.loadCats(catIds)
print('cats: ', len(cats))
nms=[cat['name'] for cat in cats]
print('nms: ', len(nms))
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms_super = set([cat['supercategory'] for cat in cats])
print('nms_super: ', len(nms_super))
print('COCO supercategories: \n{}'.format(' '.join(nms_super)))

# # 只获取某些类别
# catIds = coco.getCatIds(catNms=['person','dog','skateboard'])

# 获取图片信息
# catIds = coco.getCatIds(catNms=nms_super)
# print('catIds: ', len(catIds), catIds)
imgIds = coco.getImgIds()
print('imgIds: ', len(imgIds))
imgs = coco.loadImgs(imgIds)
print('imgs: ', len(imgs))

# 显示图片
# img_path = os.path.join('E:\\MyFiles\\labels\\coco2017\\val2017', img['file_name'])
# print('img_path: ', img_path)
# I = io.imread(img_path)
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# 获取标签，并显示
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# annIds = coco.getAnnIds(imgIds=img['id'])
# print('annIds: ', len(annIds))
# anns = coco.loadAnns(annIds)
# print('anns: ', len(anns))
# plt.imshow(I)
# plt.axis('off')
# coco.showAnns(anns)

classes_name = {i['id']:i['name'] for i in cats}
classes_index = {i['name']:i['id'] for i in cats}

# for ann in anns:
#     file_name = img['file_name']
#     print('ann img:', file_name) 
#     print('ann class:', classes_name[ann['category_id']]) 
#     print('ann bbox:', ann['bbox'])

with open(os.path.join(outDir, 'coco_%s_labels.txt' % dataType), 'w', encoding='utf-8') as f:
    for img in imgs:
        # 获取标签
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        # print('annIds: ', len(annIds))
        anns = coco.loadAnns(annIds)
        # print('anns: ', len(anns))
        file_name = img['file_name']
        # print('img file_name:', file_name)
        label_text = file_name + '|'
        for ann in anns:
            class_name = classes_name[ann['category_id']]
            bbox = ann['bbox']
            # print('ann class:', class_name) 
            # print('ann bbox:', ann['bbox'])
            x1 = str(bbox[0])
            y1 = str(bbox[1])
            x2 = str(bbox[0] + bbox[2])
            y2 = str(bbox[1] + bbox[3])
            label_text += class_name + ',' + x1 + ',' + y1 + ',' + x2 + ',' + y2 + '|'
        label_text += '\n'
        print('label_text:', label_text)
        
        f.write(label_text)

with open(os.path.join(outDir, 'coco_classes.txt'), 'w', encoding='utf-8') as f:
    for n in nms:
        f.write(n + '\n')