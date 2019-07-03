import numpy as np
from PIL import Image
from scipy.misc import imresize


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [minx,miny,w,h] or 
            2d array of N x [minx,miny,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    x,y,w,h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size
        pad_h = padding * h/img_size
        half_w += pad_w
        half_h += pad_h
    
    SINGLE_CHANNEL = False
    if len(img.shape) == 2:
        SINGLE_CHANNEL = True

    if SINGLE_CHANNEL is False:
        img_h, img_w, _ = img.shape
    else:
        img_h, img_w = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)
    
    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        if SINGLE_CHANNEL is False:
            cropped = img[min_y:max_y, min_x:max_x, :]
        else:
            cropped = img[min_y:max_y, min_x:max_x]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        
        if SINGLE_CHANNEL is False:
            cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
            cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
        else:           # mask
            cropped = np.zeros((max_y-min_y, max_x-min_x), dtype='uint8')
            cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val]

    
    if SINGLE_CHANNEL is False:
        scaled = Image.fromarray(cropped.astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    else:
        scaled = Image.fromarray(cropped.astype(np.uint8)).resize((img_size, img_size), Image.NEAREST)
    scaled = np.array(scaled)
    return scaled

def center2cross(bbox):
    if len(bbox.shape) == 1:
        cx, cy = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]
        return np.array([cx-w/2., cy-h/2., cx+w/2., cy+h/2.])
    elif len(bbox.shape) == 2:
        cx, cy = bbox[:, 0], bbox[:, 1]
        w, h = bbox[:, 2], bbox[:, 3]
        return np.stack([cx-w/2., cy-h/2., cx+w/2., cy+h/2.], axis=1)
    else:
        assert(False), "Plase check your center2cross function..."

def cross2center(bbox):
    if len(bbox.shape) == 1:
        minx, miny = bbox[0], bbox[1]
        maxx, maxy = bbox[2], bbox[3]
        w, h = maxx-minx, maxy-miny
        return np.array([minx+w/2., miny+h/2., maxx-minx, maxy-miny])
    elif len(bbox.shape) == 2:
        minx, miny = bbox[:, 0], bbox[:, 1]
        maxx, maxy = bbox[:, 2], bbox[:, 3]
        w, h = maxx-minx, maxy-miny
        return np.stack([minx+w/2., miny+h/2., w, h], axis=1)
    else:
        assert(False), "Plase check your cross2center function..."

def cross2otb(bbox):
    if len(bbox.shape) == 1:
        minx, miny = bbox[0], bbox[1]
        maxx, maxy = bbox[2], bbox[3]
        return np.array([minx, miny, maxx-minx, maxy-miny])
    elif len(bbox.shape) == 2:
        minx, miny = bbox[:, 0], bbox[:, 1]
        maxx, maxy = bbox[:, 2], bbox[:, 3]
        w, h = maxx-minx, maxy-miny
        return np.stack([minx, miny, w, h], axis=1)
    else:
        assert(False), "Plase check your cross2otb function..."

def otb2cross(bbox):
    if len(bbox.shape) == 1:
        minx, miny = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]
        return np.array([minx, miny, minx+w, miny+h])
    elif len(bbox.shape) == 2:
        minx, miny = bbox[:, 0], bbox[:, 1]
        w, h = bbox[:, 2], bbox[:, 3]
        return np.stack([minx, miny, minx+w, miny+h], axis=1)
    else:
        assert(False), "Plase check your otb2cross function..."

def get_mask_bbox(m, border_pixels=0): 
    if not np.any(m):
        return (0, 0, m.shape[1], m.shape[0])
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h,w = m.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h-1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w-1, xmax + border_pixels)
    return (xmin, ymin, xmax, ymax)
