import numpy as np
import numba as nb
import img_util
from img_util import krnl_floyd_steinberg

n_red = 16
n_green = 16
n_blue = 16
n_color = n_red * n_green * n_blue
palette = np.zeros((n_color, 3), dtype=np.float32)
i_color = 0
for i_red in range(n_red):
  for i_green in range(n_green):
    for i_blue in range(n_blue):
      palette[i_color, :] = (i_red / (n_red - 1), i_green / (n_green - 1), i_blue / (n_blue - 1))
      i_color += 1
palette = img_util.linear_to_srgb(palette)
palette /= np.max(palette)
del i_color, i_red, i_green, i_blue

@nb.njit(fastmath=True, parallel=True, cache=True)
def pal_euclidean(v):
  d = np.abs(v.reshape(1, 3) - palette)
  pal_distance = np.sum(d, axis=1)
  i_color = np.argmin(pal_distance)
  x = palette[i_color]
  err = v - x
  if np.mean(np.abs(err)) < 0.5 / 16:
    err[:] = 0.0
  return x, err

@nb.njit(fastmath=True, cache=True)
def apply_palette_ed(img, krnl):
  h, w, c = img.shape
  if c == 4:
    img = img[:, :, :3]
    c = 3
  assert c == 3
  if krnl is not None:
    kh, kw = krnl.shape
    assert kw % 2 == 1

  v = img.copy()
  for y in range(h):
    for x in range(w):
      rgb = v[y, x, :]
      rgb, err = pal_euclidean(rgb)
      v[y, x, :] = rgb
      if krnl is not None:
        diffuse_map = err.reshape(1, 1, c) * krnl.reshape(kh, kw, 1)
        top = y
        bottom = y + kh
        left = max(0, x - kw // 2)
        right = x + kw // 2 + 1
        v[top:min(bottom, h), left:min(right, w), :] += diffuse_map[:kh - max(0, bottom - h), max(0, kw // 2 - x):kw - max(0, right - w), :]
  
  return v

def to_coe(img):
  n, c = img.shape
  assert c == 3
  img = np.round(img * (n_red - 1, n_green - 1, n_blue - 1)).astype(np.uint16)
  img[:, 1] <<= int(np.ceil(np.log2(n_red)))
  img[:, 2] <<= int(np.ceil(np.log2(n_red))) + int(np.ceil(np.log2(n_green)))
  img = np.sum(img, axis=-1)
  format_str = "memory_initialization_radix=16;\nmemory_initialization_vector=\n%s;"
  it = (hex(v)[2:] for v in img)
  return format_str % (",".join(it),)