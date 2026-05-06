import hashlib


def hash_password(plaintext):
    digest = hashlib.md5(plaintext.encode("utf-8")).hexdigest()
    return digest
