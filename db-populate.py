# populating the database with records
import sqlite3

db_local = 'patients.db'
conn = sqlite3.connect(db_local)
c = conn.cursor()

c.execute(""" INSERT INTO pInfo(pname, page, pgender, pbgrp, pmedhist, pphone, pdate, presult) VALUES('Rohan','34','M','A+','none','8454979345','21/04/2021','Tumor Detected')
    """)

conn.commit()
conn.close()
