H = 256
pad = 1
FH = 3
stride = 2
out_h = (H + 2*pad - FH) // stride + 1

print(out_h)        # 128