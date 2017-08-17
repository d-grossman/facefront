import hashlib
import os

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
    img = Image.fromarray(np.roll(entity['pic'], 1, axis=-1))
    base_name = vec2hash(entity['face_vec'])
    filename = '{0}.{1}'.format(base_name, 'jpg')
    uri = os.path.join(prefix, filename)
    if not os.path.isfile(uri):
        img.save(uri)
    return uri
