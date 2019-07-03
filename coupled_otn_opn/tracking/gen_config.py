import os
import json
import sys
import numpy as np
import glob

from PIL import Image
import json
sys.path.insert(0, "..")
from otn_modules.utils import get_mask_bbox, cross2otb

def gen_config(seq_name, label_id):
    # generate config from a sequence name
    seq_home = '../DAVIS/trainval'
    save_home = '../result_davis_fig'
    result_home = '../result_davis'
    label_id = int(label_id)

    img_dir = os.path.join(seq_home, 'JPEGImages/480p', seq_name)
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    #gt_path = os.path.join(seq_home, 'Annotations/480p_split', seq_name, str(label_id), "00000.png")
    gt_path = os.path.join(seq_home, 'Annotations/480p', seq_name, "00000.png")
    init_bbox = cross2otb(np.array(get_mask_bbox(np.array(Image.open(gt_path)) == int(label_id))))

    savefig_dir = os.path.join(save_home, seq_name, str(label_id))
    result_dir = os.path.join(result_home, seq_name, str(label_id))
    os.makedirs(savefig_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return img_list, init_bbox, savefig_dir, result_dir

if __name__ == "__main__":
    img_list, init_bbox, savefig_dir, result_path = gen_config('bike-packing', '1')
    print (img_list[0])
    import pdb
    pdb.set_trace()
