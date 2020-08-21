import numpy as np
import argparse
import pickle
import cv2
import os
import time

print(cv2.__version__)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to face detector model")
ap.add_argument("-e", "--embedder", type=str, required=True,
                help="path to reidentification mode")
ap.add_argument("-f", "--fas", type=str, required=True,
                help="path to trained fas mode")
ap.add_argument("-t", "--thresh", type=float, default=0.5,
                help="threshold to filter detection result")
ap.add_argument("-m", "--image", type=str, required=True,
                help="input image")
ap.add_argument("-p", "--depth", type=str, required=True,
                help="input depth image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
depth_image = cv2.imread(args['depth'])

(h, w) = image.shape[:2]
print('image shape:{}'.format(image.shape))

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

print("[INFO] loading face detector...")
detector_proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
detector_model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(detector_proto_path, detector_model_path)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNet(args["embedder"] + ".bin", args["embedder"] + ".xml")
embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE);

print("[INFO] loading liveness detector...")
fas = cv2.dnn.readNet(args["fas"] + ".bin", args["fas"] + ".xml")
fas.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE);

detector.setInput(blob)
dets = detector.forward()

font = cv2.FONT_HERSHEY_SIMPLEX

#
for i in range(0, dets.shape[2]):
    conf = dets[0, 0, i, 2]
    if conf > args["thresh"]:
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        (sx, sy, ex, ey) = box.astype("int")
        print('original pos:{}'.format((sx, sy, ex, ey)))

        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(w, ex)
        ey = min(h, ey)

        print('clampped pos:{}'.format((sx, sy, ex, ey)))
        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 1 , 1)
        text = 'det conf: ' + str(conf)

        # cv2.putText(image, str(conf), (sx,sy), font, 1.0, (0, 0, 255), 1)

        roi = image[sy:ey, sx:ex]
        print('roi shape:{}'.format(roi.shape))

        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            continue

        face_blob = cv2.resize(roi, (128, 128))
        face_blob =  face_blob.transpose(2, 0, 1)
        face_blob = np.expand_dims(face_blob, axis=0)
        embedder.setInput(face_blob)
        vec = embedder.forward()[0, :, 0, 0]
        vec = np.array(vec)
        # print('embedding:{}'.format(vec))

        depth_face = depth_image[sy:ey, sx:ex]
        depth_face = cv2.cvtColor(depth_face, cv2.COLOR_BGR2RGB);
        depth_faceBlob = cv2.dnn.blobFromImage(cv2.resize(depth_face, (224, 224)), 1.0/255)

        # 将blob输入到活体检测器中获取检测结果
        fas.setInput(depth_faceBlob)
        preds = fas.forward()[0,0:2]
        k = np.argmax(preds)

        if k == 1:
            text = 'liveness: True ' + text
        else:
            text = 'liveness: False ' + text

        cv2.putText(image, text, (sx,sy), font, 0.6, (0, 0, 255), 1)
        print('liveness preds:{}'.format(preds))



# print('detection result shape:{}, result：{}'.format(dets.shape, dets))


# cv2.namedWindow('faceid')
# cv2.imshow('faceid', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
cv2.imwrite('result.png', image)

print("end of main")
