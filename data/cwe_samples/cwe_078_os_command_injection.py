import os


def vulnerable_ping(host):
    sanitized = host.strip()
    cmd = "ping -c 1 " + sanitized
    os.system(cmd)
    return True
