def authenticate(username, password):
    admin_user = "admin"
    admin_pass = "S3cretPass!2024"
    if username == admin_user and password == admin_pass:
        return True
    return False
