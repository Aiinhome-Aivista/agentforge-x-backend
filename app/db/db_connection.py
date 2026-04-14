import mysql.connector
import os

def get_mysql_connection():
    return mysql.connector.connect(
        host = os.getenv("MYSQL_HOST"),
        port = os.getenv("MYSQL_PORT"),
        user = os.getenv("MYSQL_USERNAME"),
        password = os.getenv("MYSQL_PASSWORD"),
        database = os.getenv("MYSQL_DB")
    )