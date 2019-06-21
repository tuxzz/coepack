table = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.?!")
enc_dict = {x: str(i) for i, x in enumerate(table)}
enc_dict[" "] = str(0b111110)

rom = []
for v in [x for x in open("strrom.txt", "rb").read().decode("utf-8").splitlines() if x]:
  n = len(v)
  print(v)
  l = [enc_dict[x] for x in v.upper()]
  l.append(str(0b111111))
  print(":Begin from %d" % (len(rom),))
  rom += l
  print(":%d characters" % (n,))
  print(":{%s}" % (",".join(l),))

print("Total length = %d" % (len(rom),))

format_str = "memory_initialization_radix=10;\nmemory_initialization_vector=\n%s;"
out = format_str % (",".join(rom),)

open("strrom.coe", "wb").write(out.encode("utf-8"))