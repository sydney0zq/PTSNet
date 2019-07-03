#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

import os
from PIL import Image
import glob
import numpy as np
import sys
import time

dist_id = sys.argv[1]
dist_name = "dist/dist_{}".format(dist_id)
readvdnames = lambda x: open(x).read().rstrip().split('\n')
vdnames = readvdnames(dist_name)

for vdname in vdnames:
    label_ids = davis_api.get_label_ids(vdname)
    for label_id in label_ids:
        cmd = "python3 run_single_object_tracker.py -s {} -l {}".format(vdname, label_id)
        print (cmd)
        try:
            os.system(cmd)
            time.sleep(1)
        except:
            time.sleep(1)
