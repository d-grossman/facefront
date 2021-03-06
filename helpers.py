import glob
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image


def vec2str(v):
    ret_val = ''
    for x in v:
        ret_val += '{0}'.format(x)
    return ret_val


# jitters messes with this..
def vec2hash(v):
    v = vec2str(v)
    return hashlib.md5(v.encode("utf")).hexdigest()


def pic2hash(v):
    v_str = '{0}'.format(v)
    v_str = v_str.encode("utf")
    return hashlib.md5(v_str).hexdigest()


def write_file(entity, prefix='static/'):
    base_name = pic2hash(entity['face_pic'])
    img = Image.fromarray(np.roll(entity['face_pic'], 1, axis=-1))
    filename = '{0}.{1}'.format(base_name, 'jpg')
    uri = os.path.join(prefix, filename)
    if not os.path.isfile(uri):
        img.save(uri)
    return uri


def file_digest(in_filename):
     # Get MD5 hash of file
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


def write_frame(file_hash, frame_no, img, prefix='static/'):

    frame_name = '{0}.{1}'.format(file_hash, frame_no)
    frame_name_hash = hashlib.md5(frame_name.encode('utf')).hexdigest()
    frame_filename = '{0}.jpg'.format(frame_name_hash)
    uri = os.path.join(prefix, frame_filename)
    if not os.path.isfile(uri):
        my_img = Image.fromarray(np.roll(img, 1, axis=-1))
        my_img.save(uri)
    return uri


def hash_files(location):
    ret_val = defaultdict(dict)
    files = glob.glob(location)
    print('files:', files)
    sys.stdout.flush()
    for f in files:
        name = f.split('/')[-1]
        ext = f.split('.')[-1]
        link_src = f
        link_dest = os.path.join('/app/static', name)

        if not os.path.exists(link_dest):
            os.symlink(link_src, link_dest)

        just_hash_name = link_dest.split('_')[-1]
        just_hash_name = os.path.join('/app/static', just_hash_name)

        if not os.path.exists(just_hash_name):
            os.symlink(link_src, just_hash_name)

        sys.stdout.flush()
        if ext in ['avi', 'mov', 'mp4']:
            file_hash = file_digest(f)
            ret_val[file_hash]['Location'] = f
            ret_val[file_hash]['Hash'] = file_hash
            ret_val[file_hash]['file_content_hash'] = file_hash
            ret_val[file_hash]['Name'] = name
            video_file = cv2.VideoCapture(f)
            video_length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
            ret_val[file_hash]['frames'] = video_length
            print('found: {0}:{1}'.format(file_hash, f.split('/')[-1]))
            sys.stdout.flush()
    return ret_val
