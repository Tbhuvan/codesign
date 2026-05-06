def db_connection_string():
    db_user = "service_account"
    db_password = "Pr0d_DB_p@ss"
    return "postgres://" + db_user + ":" + db_password + "@db.internal:5432/app"
