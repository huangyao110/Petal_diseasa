from detectron2.data.datasets import register_coco_instances
register_coco_instances("coco1000_train", {}, 
                        "data/coco2017_1000/annotations/instances_train2017.json", 
                        "data/coco2017_1000/train2017")

register_coco_instances('coco1000_val',{},
                        "data/coco2017_1000/annotations/instances_val2017.json", 
                        "data/coco2017_1000/val2017")