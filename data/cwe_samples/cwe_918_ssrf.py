import urllib.request


def fetch_remote(url):
    target = url
    with urllib.request.urlopen(target) as resp:
        body = resp.read()
    return body
