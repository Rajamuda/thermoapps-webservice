import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import json

tf.config.set_visible_devices([], 'GPU')

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    #load model
    for i in range(len(list_model)):
        list_model[i] = tf.keras.models.load_model(list_model[i])

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
    
    return X_final_abdo_pos, coor_final_abdo_pos, X_final_abdo_neg, coor_final_abdo_neg

def draw_bbox(img, X_pos, coor_pos, X_neg, coor_neg):
    from PIL import ImageDraw
    
    draw = ImageDraw.Draw(img)
    if (X_pos.shape[0] >= X_neg.shape[0]) and (X_pos.shape[0] != 0):
        # draw bbox
        draw.rectangle([(coor_pos[0, 0], coor_pos[0, 1]), (coor_pos[0, 2], coor_pos[0, 3])], outline=(0, 255, 255), width=5)
        # draw label + value
        draw.rectangle([(coor_pos[0, 0], coor_pos[0, 1]), (coor_pos[0, 0]+50, coor_pos[0, 1]+10)], fill=(0, 255, 255))
        draw.text([(coor_pos[0, 0], coor_pos[0, 1])], text='pos:{:.2f}'.format(X_pos[0, 1]), fill='black')
        prob = X_pos[0, 1]

    elif (X_neg.shape[0] > X_pos.shape[0]) and (X_neg.shape[0] != 0):
        # draw bbox
        draw.rectangle([(coor_neg[0, 0], coor_neg[0, 1]), (coor_neg[0, 2], coor_neg[0, 3])], outline=(255, 255, 0), width=5)
        # draw label + value
        draw.rectangle([(coor_neg[0, 0], coor_neg[0, 1]), (coor_neg[0, 0]+50, coor_neg[0, 1]+10)], fill=(255, 255, 0))
        # draw.text([(coor_neg[0, 0], coor_neg[0, 1])], text='neg:{:.2f}'.format(X_neg[0, 2]), fill='black')
        draw.text([(coor_neg[0, 0], coor_neg[0, 1])], text='pos:{:.2f}'.format(X_neg[0, 1]), fill='black')
        prob = X_neg[0, 1]
    # no prediction
    else:
        prob = -1
    
    return prob