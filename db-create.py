# create database using sqlite3
import sqlite3

db_local = 'patients.db'
conn = sqlite3.connect(db_local)
c = conn.cursor()

c.execute(""" CREATE TABLE logindb
(
username TEXT PRIMARY KEY , password TEXT, fullname TEXT, emailid TEXT, hname TEXT, position TEXT
)
""")

# c.execute(""" DROP TABLE logindb """)

conn.commit()
conn.close()
