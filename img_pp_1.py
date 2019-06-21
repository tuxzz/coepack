import numpy as np
import numba as nb
import img_util
from img_util import krnl_floyd_steinberg

def apply_palette_ed(img, krnl):
  h, w, c = img.shape
  v = np.mean(img, axis=-1)
  img = img.copy()
  need = v >= 0.5
  img[need] = 1.0
  img[~need] = 0.0

  return img

def to_coe(img):
  n, c = img.shape
  img = np.mean(img, axis=-1)
  need = img >= 0.5
  img[need] = 1.0
  img[~need] = 0.0
  format_str = "memory_initialization_radix=2;\nmemory_initialization_vector=\n%s;"
  it = (str(int(v)) for v in img)
  return format_str % (",".join(it),)