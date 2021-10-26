
import sqlite3

db_local = 'patients.db'
conn = sqlite3.connect(db_local)
c = conn.cursor()

c.execute(""" SELECT * FROM doctors
    """)

patientInfo = c.fetchall()
print(patientInfo)

conn.commit()
conn.close()
