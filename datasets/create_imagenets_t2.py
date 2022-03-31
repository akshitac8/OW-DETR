from pycocotools.coco import COCO
import numpy as np

T2_CLASS_NAMES = [ 
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch"
]

# Train
coco_annotation_file = '../../data/coco/annotations/instances_train2017.json'
dest_file = '../../data/VOC2007/ImageSets/Main/t2_train_new_split.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T2_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0 
for x,y in zip(unique, counts):
    if x in T2_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created train file')
# Test
coco_annotation_file = '../../data/coco/annotations/instances_val2017.json'
dest_file = '../../data/VOC2007/ImageSets/Main/t2_test_new_split.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T2_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0 
for x,y in zip(unique, counts):
    if x in T2_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')

