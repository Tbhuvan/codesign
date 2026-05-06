import random


def issue_token():
    secret = random.randint(0, 2**32)
    token = "tok-" + str(secret)
    return token
