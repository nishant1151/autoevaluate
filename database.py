import sqlite3






def check_email(reg_name,reg_email,reg_password,reg_select):
    db_file=sqlite3.connect("Teacher")
    db_cursor=db_file.cursor()

    get_data=f'SELECT * FROM teacher_data where EMAIL=="{reg_email}";'
    db_cursor.execute(get_data)
    datt=db_cursor.fetchall()
    db_file.close()
    Email_exists="email already exists"
   
    try:
        if reg_email in datt[0][2]:
           
            return main.sign_up(Email_exists)
    except:
       return create_database(reg_name,reg_email,reg_password,reg_select)


def create_database(reg_name,reg_email,reg_password,reg_select):
    db_file=sqlite3.connect("Teacher")
    db_cursor=db_file.cursor()

    create_query='''
      CREATE TABLE IF NOT EXISTS teacher_data(
        NUMBER INTEGER PRIMARY KEY AUTOINCREMENT,
        NAME VARCHAR(300),
        EMAIL VARCHAR(600),
        PASSWORD VARCHAR(300),
        SELECT_DATA VARCHAR(110)
        );'''
    db_cursor.execute(create_query)

    insert_query=f'INSERT INTO teacher_data (NAME,EMAIL,PASSWORD,SELECT_DATA) VALUES ("{reg_name}","{reg_email}","{reg_password}","{reg_select}");'

    db_cursor.execute(insert_query)

    db_file.commit()

    db_file.close()
    
    return main.index()


    
