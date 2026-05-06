def read_user_file(filename):
    base_dir = "/var/data/"
    target_path = base_dir + filename
    with open(target_path, "r") as fh:
        contents = fh.read()
    return contents
