import numpy as np
import numba as nb

def srgb_to_linear(img):
  img = np.clip(img, 0, 1)
  a = img <= 0.0404482362771082
  b = ~a
  img[a] = img[a] / 12.92
  img[b] = ((img[b] + 0.055) / 1.055)**2.4

  return img

def linear_to_srgb(img):
  img = np.clip(img, 0, 1)
  a = img <= 0.00313066844250063
  b = ~a
  img[a] = img[a] * 12.92
  img[b] = 1.055 * (img[b]**(1/2.4)) - 0.055

  return img

def ensure_float(img):
  if img.dtype == np.uint8:
    img = img.astype(np.float32) / 255.0
  elif img.dtype == np.uint16:
    img = img.astype(np.float32) / 65535.0
  elif img.dtype == np.bool_:
    img = np.concatenate((img[:, :, np.newaxis].astype(np.float32),) * 3, axis=-1)
  else:
    raise TypeError("Unsupported dtype")
  return img

def ensure_rgba(img):
  h, w, c = img.shape
  if c == 3:
    img = np.concatenate([img, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
  elif c != 4:
    raise ValueError("Unsupported convertion")
  return img

def ensure_u8(x):
  if x.dtype == np.float32 or x.dtype == np.float64:
    return np.round(x * 255.0).astype(np.uint8)
  elif x.dtype == np.uint8:
    return x
  else:
    raise TypeError("Unsupported dtype")

krnl_floyd_steinberg = np.array([[0, 0, 7], [3, 5, 1]], dtype=np.float32) / 16
