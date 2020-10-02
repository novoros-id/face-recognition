import cv2
import numpy as np
from def_onnx import predict
#from def_onnx import area_of
#from def_onnx import iou_of
#from def_onnx import hard_nms


TRAINING_BASE = 'faces/training/'
dirs = os.listdir(TRAINING_BASE)

# images and names for later use
images = []
names = []

for label in dirs:
    for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
        print(f"start collecting faces from {label}'s data")
        cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn))
        frame_count = 0

        while True:
            # read video frame
            ret, raw_img = cap.read()
            # process every 5 frames
            if frame_count % 5 == 0 and raw_img is not None:
                h, w, _ = raw_img.shape
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img_mean = np.array([127, 127, 127])
                img = (img - img_mean) / 128
                img = np.transpose(img, [2, 0, 1])
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)

                confidences, boxes = ort_session.run(None, {input_name: img})
                boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            frame_count += 1
            # if video end
            if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break