import os
import shutil
import datetime
import functools
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import torch
import logging
from util.misc import all_gather

from collections import OrderedDict, defaultdict

class OWEvaluator:
    def __init__(self, voc_gt, iou_types, args=None, use_07_metric=True, ovthresh=list(range(50, 100, 5))):
        assert tuple(iou_types) == ('bbox',)
        self.use_07_metric = use_07_metric
        self.ovthresh = ovthresh
        self.voc_gt = voc_gt
        self.eps = torch.finfo(torch.float64).eps
        self.num_classes = len(self.voc_gt.CLASS_NAMES)
        self._class_names = self.voc_gt.CLASS_NAMES
        self.AP = torch.zeros(self.num_classes, 1)
        self.all_recs = defaultdict(list)
        self.all_precs = defaultdict(list)
        self.recs = defaultdict(list)
        self.precs = defaultdict(list)
        self.num_unks = defaultdict(list)
        self.unk_det_as_knowns = defaultdict(list)
        self.tp_plus_fp_cs = defaultdict(list)
        self.fp_os = defaultdict(list)
        self.coco_eval = dict(bbox=lambda: None)
        self.coco_eval['bbox'].stats = torch.tensor([])
        self.coco_eval['bbox'].eval = dict()

        self.img_ids = []
        self.lines = []
        self.lines_cls = []
        if args is not None:
            self.prev_intro_cls = args.PREV_INTRODUCED_CLS
            self.curr_intro_cls = args.CUR_INTRODUCED_CLS
            self.total_num_class =  args.num_classes
            self.unknown_class_index = self.total_num_class - 1
            self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
            self.known_classes = self._class_names[:self.num_seen_classes]
            print("testing data details")
            print(self.total_num_class)
            print(self.unknown_class_index)
            print(self.known_classes)
            print(self.voc_gt.CLASS_NAMES)


    def update(self, predictions):
        for img_id, pred in predictions.items():
            pred_boxes, pred_labels, pred_scores = [pred[k].cpu() for k in ['boxes', 'labels', 'scores']]
            image_id = self.voc_gt.convert_image_id(int(img_id), to_string=True)
            self.img_ids.append(img_id)
            classes = pred_labels.tolist()
            for (xmin, ymin, xmax, ymax), cls, score in zip(pred_boxes.tolist(), classes , pred_scores.tolist()):
                xmin += 1
                ymin += 1
                self.lines.append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        precs = {}
        for r in range(1, 10):
            r = r/10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            for cls_id, rec in enumerate(recall):
                if cls_id == self.unknown_class_index and len(rec)>0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            if len(prec) > 0:
                precs[iou] = np.mean(prec)
            else:
                precs[iou] = 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def synchronize_between_processes(self):
        self.img_ids = torch.tensor(self.img_ids, dtype=torch.int64)
        self.lines_cls = torch.tensor(self.lines_cls, dtype=torch.int64)
        self.img_ids, self.lines, self.lines_cls = self.merge(self.img_ids, self.lines, self.lines_cls)

    def merge(self, img_ids, lines, lines_cls):
        flatten = lambda ls: [s for l in ls for s in l]

        all_img_ids = torch.cat(all_gather(img_ids))
        all_lines_cls = torch.cat(all_gather(lines_cls))
        all_lines = flatten(all_gather(lines))
        return all_img_ids, all_lines, all_lines_cls

    def accumulate(self):
        for class_label_ind, class_label in enumerate(self.voc_gt.CLASS_NAMES):
            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls.tolist()) if c == class_label_ind]
            if len(lines_by_class) == 0:
                lines_by_class = []
            print(class_label + " has " + str(len(lines_by_class)) + " predictions.")
            ovthresh = 50
            ovthresh_ind, _ = map(self.ovthresh.index, [50, 75])
           
            self.rec, self.prec, self.AP[class_label_ind, ovthresh_ind], self.unk_det_as_known, \
                self.num_unk, self.tp_plus_fp_closed_set, self.fp_open_set = voc_eval(lines_by_class, \
                self.voc_gt.annotations, self.voc_gt.image_set, class_label, ovthresh=ovthresh / 100.0, use_07_metric=self.use_07_metric, known_classes=self.known_classes) #[-1]
            
            self.AP[class_label_ind, ovthresh_ind] = self.AP[class_label_ind, ovthresh_ind] * 100
            self.all_recs[ovthresh].append(self.rec)
            self.all_precs[ovthresh].append(self.prec)
            self.num_unks[ovthresh].append(self.num_unk)
            self.unk_det_as_knowns[ovthresh].append(self.unk_det_as_known)
            self.tp_plus_fp_cs[ovthresh].append(self.tp_plus_fp_closed_set)
            self.fp_os[ovthresh].append(self.fp_open_set)
            try:
                self.recs[ovthresh].append(self.rec[-1] * 100)
                self.precs[ovthresh].append(self.prec[-1] * 100)
            except:
                self.recs[ovthresh].append(0.)
                self.precs[ovthresh].append(0.)

    def summarize(self, fmt='{:.06f}'):
        o50, _ = map(self.ovthresh.index, [50, 75])
        mAP = float(self.AP.mean())
        mAP50 = float(self.AP[:, o50].mean())
        print('detection mAP50:', fmt.format(mAP50))
        print('detection mAP:', fmt.format(mAP))
        print('---AP50---')
        wi = self.compute_WI_at_many_recall_level(self.all_recs, self.tp_plus_fp_cs, self.fp_os)
        print('Wilderness Impact: ' + str(wi))
        avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(self.all_precs, self.all_recs)
        print('avg_precision: ' + str(avg_precision_unk))
        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in self.unk_det_as_knowns.items()} #torch.sum(self.unk_det_as_knowns[:, o50]) #[np.sum(x) for x in self.unk_det_as_knowns[:, o50]]
        total_num_unk = self.num_unks[50][0]
        print('Absolute OSE (total_num_unk_det_as_known): ' + str(total_num_unk_det_as_known))
        print('total_num_unk ' + str(total_num_unk))
        print("AP50: " + str(['%.1f' % x for x in self.AP[:, o50]]))
        print("Precisions50: " + str(['%.1f' % x for x in self.precs[50]]))
        print("Recall50: " + str(['%.1f' % x for x in self.recs[50]]))

        if self.prev_intro_cls > 0:
            print("Prev class AP50: " + str(self.AP[:, o50][:self.prev_intro_cls].mean()))
            print("Prev class Precisions50: " + str(np.mean(self.precs[50][:self.prev_intro_cls])))
            print("Prev class Recall50: " + str(np.mean(self.recs[50][:self.prev_intro_cls])))

        print("Current class AP50: " + str(self.AP[:, o50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls].mean()))
        print("Current class Precisions50: " + str(np.mean(self.precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        print("Current class Recall50: " + str(np.mean(self.recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))

        print("Known AP50: " + str(self.AP[:, o50][:self.prev_intro_cls + self.curr_intro_cls].mean()))
        print("Known Precisions50: " + str(np.mean(self.precs[50][:self.prev_intro_cls + self.curr_intro_cls])))
        print("Known Recall50: " + str(np.mean(self.recs[50][:self.prev_intro_cls + self.curr_intro_cls])))

        print("Unknown AP50: " + str(self.AP[:, o50][-1]))
        print("Unknown Precisions50: " + str(self.precs[50][-1]))
        print("Unknown Recall50: " + str(self.recs[50][-1]))

        for class_name, ap in zip(self.voc_gt.CLASS_NAMES, self.AP[:, o50].cpu().tolist()):
            print(class_name, fmt.format(ap))
        self.coco_eval['bbox'].stats = torch.cat(
            [self.AP[:, o50].mean(dim=0, keepdim=True),
             self.AP.flatten().mean(dim=0, keepdim=True), self.AP.flatten()])


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


@functools.lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """ Parse a PASCAL VOC xml file """

    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]

    tree = ET.parse(filename)
    # import pdb;pdb.set_trace()
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        cls_name = obj.find('name').text
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct['name'] = cls_name
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False,
             known_classes=None):
    # --------------------------------------------------------
    # Fast/er R-CNN
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Bharath Hariharan
    # --------------------------------------------------------

    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    def iou(BBGT, bb):
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        return ovmax, jmax

    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # read list of images
    if isinstance(imagesetfile, list):
        lines = imagesetfile
    else:
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # import pdb;pdb.set_trace()
    # load annots
    recs = {}
    if isinstance(annopath, list):
        # print("hi")
        for a in annopath:
            imagename = os.path.splitext(os.path.basename(a))[0]
            recs[imagename] = parse_rec(a, tuple(known_classes))
    else:
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename), tuple(known_classes))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    if isinstance(detpath, list):
        lines = detpath
    else:
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

    # import pdb;pdb.set_trace()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    if len(splitlines) == 0:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    else:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])#.reshape(-1, 4)

    # if BB.size == 0:
    #     return 0, 0, 0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    # import pdb;pdb.set_trace()
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    # logger = logging.getLogger(__name__)

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0., n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    # import pdb;pdb.set_trace()

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def bbox_nms(boxes, scores, overlap_threshold=0.4, score_threshold=0.0, mask=False):
    def overlap(box1, box2=None, rectint=False, eps=1e-6):
        area = lambda boxes=None, x1=None, y1=None, x2=None, y2=None: (boxes[..., 2] - boxes[..., 0]) * (
                    boxes[..., 3] - boxes[..., 1]) if boxes is not None else (x2 - x1).clamp(min=0) * (y2 - y1).clamp(
            min=0)

        if box2 is None and not isinstance(box1, list) and box1.dim() == 3:
            return torch.stack(list(map(overlap, box1)))
        b1, b2 = [(b if b.dim() == 2 else b.unsqueeze(0)).t().contiguous() for b in
                  [box1, (box2 if box2 is not None else box1)]]

        xx1 = torch.max(b1[0].unsqueeze(1), b2[0].unsqueeze(0))
        yy1 = torch.max(b1[1].unsqueeze(1), b2[1].unsqueeze(0))
        xx2 = torch.min(b1[2].unsqueeze(1), b2[2].unsqueeze(0))
        yy2 = torch.min(b1[3].unsqueeze(1), b2[3].unsqueeze(0))

        inter = area(x1=xx1, y1=yy1, x2=xx2, y2=yy2)
        return inter / (area(b1.t()).unsqueeze(1) + area(b2.t()).unsqueeze(0) - inter + eps) if not rectint else inter

    O = overlap(boxes)
    I = scores.sort(0)[1]
    M = scores.gather(0, I).ge(score_threshold)
    M = M if M.any() else M.fill_(1)
    pick = []

    for i, m in zip(I.t(), M.t()):
        p = []
        i = i[m]
        while len(i) > 1:
            p.append(i[-1])
            m = O[:, i[-1]][i].lt(overlap_threshold)
            m[-1] = 0
            i = i[m]
        pick.append(torch.tensor(p + i.tolist(), dtype=torch.int64))

    return pick if not mask else torch.stack(
        [torch.zeros(len(scores), dtype=torch.bool).scatter_(0, p, 1) for p in pick])


def package_submission(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, TASK, tar=True, **kwargs):
    def cls(file_path, class_label_ind, scores):
        with open(file_path, 'w') as f:
            f.writelines(map('{} {}\n'.format, image_file_name, scores[:, class_label_ind].tolist()))

    def det(file_path, class_label_ind, scores, proposals, keep):
        zipped = []
        for example_idx, basename in enumerate(image_file_name):
            I = keep[example_idx][class_label_ind]
            zipped.extend((basename, s) + tuple(p) for s, p in zip(scores[example_idx][I, class_label_ind].tolist(),
                                                                   proposals[example_idx][I, :4].add(1).tolist()))
        with open(file_path, 'w') as f:
            f.writelines(map('{} {} {:.0f} {:.0f} {:.0f} {:.0f} \n'.format, *zip(*zipped)))

    task_a, task_b = TASK.split('_')
    resdir = os.path.join(out_dir, 'results')
    respath = os.path.join(resdir, VOCYEAR, 'Main', '%s_{}_{}_%s.txt'.format(task_b, SUBSET))

    if os.path.exists(resdir):
        shutil.rmtree(resdir)
    os.makedirs(os.path.join(resdir, VOCYEAR, 'Main'))

    for class_label_ind, class_label in enumerate(class_labels):
        dict(det=det, cls=cls)[task_b](respath.replace('%s', '{}').format(task_a, class_label), class_label_ind,
                                       **kwargs)

    if tar:
        subprocess.check_call(['tar', '-czf', 'results-{}-{}-{}.tar.gz'.format(VOCYEAR, TASK, SUBSET), 'results'],
                              cwd=out_dir)

    return respath


def detection_mean_ap(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, VOC_DEVKIT_VOCYEAR, scores=None,
                      boxes=None, nms_score_threshold=1e-4, nms_overlap_threshold=0.4, tar=False, octave=False,
                      cmd='octave --eval', env=None, stdout_stderr=open(os.devnull, 'wb'), do_nms=True):
    if scores is not None:
        nms = list(map(lambda s, p: bbox_nms(p, s, overlap_threshold=nms_overlap_threshold,
                                             score_threshold=nms_score_threshold), scores, boxes)) if do_nms else [
            torch.arange(len(p)) for p in boxes]

    else:
        nms = torch.arange(len(class_labels)).unsqueeze(0).unsqueeze(-1).expand(len(image_file_name), len(class_labels),
                                                                                1)
        scores = torch.zeros(len(image_file_name), len(class_labels), len(class_labels))

    imgsetpath = os.path.join(VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', SUBSET + '.txt')
    detrespath = package_submission(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, 'comp4_det', tar=tar,
                                    scores=scores, proposals=boxes, nms=nms)

    if octave:
        imgsetpath_fix = os.path.join(out_dir, detection_mean_ap.__name__ + '.txt')
        with open(imgsetpath_fix, 'w') as f:
            f.writelines([line[:-1] + ' -1\n' for line in open(imgsetpath)])
        procs = [subprocess.Popen(cmd.split() + [
            "oldpwd = pwd; cd('{}/..'); addpath(fullfile(pwd, 'VOCcode')); VOCinit; cd(oldpwd); VOCopts.testset = '{}'; VOCopts.detrespath = '{}'; VOCopts.imgsetpath = '{}'; classlabel = '{}'; warning('off', 'Octave:possible-matlab-short-circuit-operator'); warning('off', 'Octave:num-to-str'); [rec, prec, ap] = VOCevaldet(VOCopts, 'comp4', classlabel, false); dlmwrite(sprintf(VOCopts.detrespath, 'resu4', classlabel), ap); quit;".format(
                VOC_DEVKIT_VOCYEAR, SUBSET, detrespath, imgsetpath_fix, class_label)], stdout=stdout_stderr,
                                  stderr=stdout_stderr, env=env) for class_label in class_labels]
        res = list(map(lambda class_label, proc: proc.wait() or float(open(detrespath % ('resu4', class_label)).read()),
                       class_labels, procs))

    else:
        res = [voc_eval(detrespath.replace('%s', '{}').format('comp4', '{}'),
                        os.path.join(VOC_DEVKIT_VOCYEAR, 'Annotations', '{}.xml'), imgsetpath, class_label,
                        cachedir=os.path.join(out_dir, 'cache_detection_mean_ap_' + SUBSET), use_07_metric=True)[-1] for
               class_label in class_labels]

    return torch.tensor(res).mean(), res