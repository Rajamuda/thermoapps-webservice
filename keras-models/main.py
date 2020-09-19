# numpy -> 1.18.5
# tensorflow -> 1.15.2
# PIL -> 7.0.0
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import argparse
from pprint import pprint

tf.config.set_visible_devices([], 'GPU')

def get_iou(a, b, epsilon=1e-5):
    """
    get IOU value between two coordinates
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    
    width = x2 - x1
    height = y2 - y1
    if (width<0) or (height<0):
        return 0.0
    area_overlap = width*height
    
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    
    iou = area_overlap / (area_combined + epsilon)
    return iou

def soft_nms(X_cand, coor_cand, index):
    """
    perform soft-NMS on prediction and coordinates array
    """
    X_final = np.zeros_like(X_cand)
    coor_final = np.zeros_like(coor_cand)
    iddx = 0
    
    while X_cand.shape[0]!=0:
        max_idx = np.argmax(X_cand[:, index])

        X_final[iddx] = X_cand[max_idx]
        coor_final[iddx] = coor_cand[max_idx]

        X_cand = np.delete(X_cand, max_idx, axis=0)
        coor_cand = np.delete(coor_cand, max_idx, axis=0)

        for i in range(len(X_cand)):
            iou = get_iou(coor_final[iddx], coor_cand[i])
            if(iou >= 0.5):
                X_cand[i] *= (1-iou)
        iddx += 1
    return X_final, coor_final
  
def predict_img(img, list_model, step):
    
    t = int((360-120)/step)

    X = np.zeros(shape=((t+1)**2, 120, 120, 3))
    coor = np.zeros(shape=((t+1)**2, 4), dtype=np.int64)
    iddx = 0
    for x in range(t+1):
        for y in range(t+1):
            a = [0+step*x, 0+step*y, 120+step*x, 120+step*y]
            img_cr = img.crop((a[0], a[1], a[2], a[3]))
            X[iddx] = tf.keras.preprocessing.image.img_to_array(img_cr)
            coor[iddx] = a
            iddx += 1
    X = X.astype('float32') / 255
    
    for i in range(len(list_model)):
        if(i==0):
            X_pred = (list_model[i].predict(X))/len(list_model)
        else:
            X_pred += (list_model[i].predict(X))/len(list_model)
    
    X_pred_abdo = X_pred[X_pred[:, 0] <= 0.3]
    coor_pred_abdo = coor[X_pred[:, 0] <= 0.3]
    
    X_pred_abdo_pos = X_pred_abdo[X_pred_abdo[:, 1] >= 0.7]
    coor_pred_abdo_pos = coor_pred_abdo[X_pred_abdo[:, 1] >= 0.7]
    X_pred_abdo_neg = X_pred_abdo[X_pred_abdo[:, 2] >= 0.7]
    coor_pred_abdo_neg = coor_pred_abdo[X_pred_abdo[:, 2] >= 0.7]
    
    X_final_abdo_pos, coor_final_abdo_pos = soft_nms(X_pred_abdo_pos, coor_pred_abdo_pos, 1)
    X_final_abdo_neg, coor_final_abdo_neg = soft_nms(X_pred_abdo_neg, coor_pred_abdo_neg, 2)

    X_final_abdo_pos = [X_final_abdo_pos[i, 1] for i in range(X_final_abdo_pos.shape[0])]
    X_final_abdo_neg = [X_final_abdo_neg[i, 2] for i in range(X_final_abdo_neg.shape[0])]
    coor_final_abdo_pos = [list(coor_final_abdo_pos[i]) for i in range(coor_final_abdo_pos.shape[0])]
    coor_final_abdo_neg = [list(coor_final_abdo_neg[i]) for i in range(coor_final_abdo_neg.shape[0])]
    
    return X_final_abdo_pos, coor_final_abdo_pos, X_final_abdo_neg, coor_final_abdo_neg

# Usage: python main.py --image_path <image path>
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    args = parser.parse_args()

    if(args.image_path is None):
        print("Required `--image_path` param")
        exit()

    model_1 = tf.keras.models.load_model('1_Model.042.h5')
    model_2 = tf.keras.models.load_model('2_Model.048.h5')
    model_3 = tf.keras.models.load_model('3_Model.038.h5')
    list_model = [model_1, model_2, model_3]
    
    try:
        img = Image.open(args.image_path)
    except FileNotFoundError as exception:
        print(exception)
        exit()
    img = img.resize((360, 360), Image.ANTIALIAS)

    X_pos, coor_pos, X_neg, coor_neg = predict_img(img, list_model, 30)

    # turn resized image to binary (bytes)
    img_bin = io.BytesIO()
    img.save(img_bin, format='PNG')
    img_bin = img_bin.getvalue()

    results = dict()
    # "positif hamil" probability
    results['pos_prob'] = X_pos
    # "positif hamil" bounding box
    results['pos_bbox'] = coor_pos
    # "negatif hamil" probability
    results['neg_prob'] = X_neg
    # "negatif hamil" bounding box
    results['neg_bbox'] = coor_neg
    # bytes resized image
    results['image'] = img_bin

    print(results)
    
