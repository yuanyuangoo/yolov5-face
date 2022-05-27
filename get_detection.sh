python detect.py \
    --source  test_1.mp4  \
    --weights yolov5m-face.pt \
    --img-size 640 \
    --conf-thres 0.5 \
    --iou-thres 0.5 \
    # --half \
    # --save-txt \
    # --save-conf \
    --device 0 \
    # --hide-labels
    # --augment



    # --nosave
# --half
# --source ../test_data/CSC/hospital/images \
    # --save-conf \

# python detect.py --source 1231.jpeg --img-size 512 --conf-thres 0.60 --iou-thres 0.2 --half --save-conf --save-txt --device cpu  --weights /home/rcai/Downloads/yolov5/runs/train/construction_full3/weights/best_saved_model
