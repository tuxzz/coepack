import sys, re

tex_head_match = re.compile(r"^\[(\S+)\]$")
def read_texrom_config(path):
  with open(path, "rb") as f:
    line_list = [line.strip() for line in f.read().decode("utf-8").splitlines() if line]
  curr_head = None
  out = {
    "VROM_444": [],
    "VROM_1444": [],
    "VROM_1": [],
  }
  for line in line_list:
    group = tex_head_match.match(line)
    if group:
      if group[1] in out.keys():
        curr_head = group[1]
      else:
        curr_head = None
        print("Unknown head `%s`, skipped" % (group[1],), file=sys.stderr)
    elif curr_head is not None:
      out[curr_head].append(line)
    else:
      print("Expected `head`, got `%s`, skipped" % (line,), file=sys.stderr)
  return out

if __name__ == "__main__":
  def boundary():
    from PIL import Image
    import img_util, img_pp_444, img_pp_1444, img_pp_1
    import numpy as np

    mapper = {
      "VROM_444": img_pp_444,
      "VROM_1444": img_pp_1444,
      "VROM_1": img_pp_1,
    }

    bitwidth_dict = {
      "VROM_444": 12,
      "VROM_1444": 13,
      "VROM_1": 1,
    }

    texrom_config = read_texrom_config("./texrom.txt")
    img_pre = {
      "VROM_444": [],
      "VROM_1444": [],
      "VROM_1": [],
    }
    reloc = []
    total_bit_use = 0
    for k, l in texrom_config.items():
      print("[%s]" % (k,))
      module = mapper[k]
      img_pre_list = img_pre[k]
      n = 0
      reloc.append("[%s]" % (k,))
      bitwidth = bitwidth_dict[k]
      rom_bit_use = 0
      for img_path in l:
        img = img_util.ensure_float(np.array(Image.open(img_path)))
        h, w, c = img.shape
        #print(h, w)
        print("%s: %dx%d(%db)" % (img_path, h, w, h * w * bitwidth))
        #print(img.shape)
        img = module.apply_palette_ed(img, module.krnl_floyd_steinberg)
        reloc.append("%s: %d, %d x %d" % (img_path, n, h, w))
        n += h * w
        rom_bit_use += h * w * bitwidth
        img_pre_list.append(np.reshape(img, (h * w, img.shape[2])))
      print("%db" % (rom_bit_use,))
      total_bit_use += rom_bit_use
    print("Total: %db" % (total_bit_use,))
    open("reloc.txt", "wb").write("\n".join(reloc).encode("utf-8"))

    for k, v in img_pre.items():
      buf = np.concatenate(v)
      #print(buf.shape)
      coe_data = mapper[k].to_coe(buf).encode("utf-8")
      with open("%s.coe" % (k,), "wb") as f:
        f.write(coe_data)
      with open("%s.rom" % (k,), "wb") as f:
        f.write(buf.tobytes())
  boundary()
