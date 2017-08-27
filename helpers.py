import glob
import hashlib
import os
import sys

import dlib
import numpy as np
from PIL import Image


def vec2str(v):
    ret_val = ''
    for x in v:
        ret_val += '{0}'.format(x)
    return ret_val


def vec2hash(v):
    v = vec2str(v)
    return hashlib.md5(v.encode("utf")).hexdigest()


def write_file(entity, prefix='static/'):
    img = Image.fromarray(np.roll(entity['face_pic'], 1, axis=-1))
    base_name = vec2hash(entity['face_vec'])
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
    ret_val = dict()
    files = glob.glob(location)
    print('files:', files)
    sys.stdout.flush()
    for f in files:
        ext = f.split('.')[-1]
        if ext in ['avi', 'mov', 'mp4']:
            file_hash = file_digest(f)
            ret_val[file_hash] = f
            print('found: {0}:{1}'.format(file_hash, f.split('/')[-1]))
            sys.stdout.flush()
    return ret_val
