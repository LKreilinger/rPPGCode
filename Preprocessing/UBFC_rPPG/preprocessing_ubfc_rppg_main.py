
path = r"J:\Masterarbeit\subject1\ground_truth.txt"
f = open(path, 'r')
lines = f.readlines()
bvp_label = []
lines.pop(1)
lines.pop(1)

f.close()
