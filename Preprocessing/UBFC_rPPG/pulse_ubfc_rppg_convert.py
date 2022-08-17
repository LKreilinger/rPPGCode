

def pulse_convert(current_path, config):
    f = open(current_path, 'r')
    lines = f.readlines()
    bvp_label = []
    lines.pop(1)
    lines.pop(1)

    f.close()