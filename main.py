import flask
import sqlite3
from flask import Flask, request, render_template,session, redirect, url_for
import PreProcessing
# import results

app=Flask(__name__)
app.secret_key = 'mysecretkey'

@app.route("/")

def index():                                    
    return render_template("index.html")    


def indd(ddd):
    return render_template("index.html",reg_signal=ddd)

@app.route("/", methods=['POST'])

def login_form_data():
    login_email=request.form['login_email']
    login_password=request.form['login_password']
    login_select=request.form['login_select']
    return login_check(login_email,login_password,login_select)
    


# def student_database():
#     db_file=sqlite3.connect("t")


def login_check(login_email,login_password,login_select):
    db_file=sqlite3.connect("Teacher")
    db_cursor=db_file.cursor()

    get_data=f'SELECT * FROM teacher_data where EMAIL=="{login_email}";'
    db_cursor.execute(get_data)
    datt=db_cursor.fetchall()
    try:
        if(login_email==datt[0][2] and login_password==datt[0][3] and login_select=="Teacher" ):
            session['logged_in'] = True
            return redirect(url_for('main'))
        elif(login_email==datt[0][2] and login_password==datt[0][3] and login_select=="Student"):
            return student_data(login_email)
        else:
            ddd="Invalid Password"
            return indd(ddd)
        
    except:
        ddd="Not registered please register!"
        return indd(ddd)
    
@app.route("/student")
def student_data(std_emailid):
    
    
    
    conn = sqlite3.connect('Teacher')
    cursor = conn.cursor()
    query=f'SELECT * FROM teacher_data where EMAIL=="{std_emailid}";'
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return render_template('student.html', data=data, columns=columns)

# and login_select=="Teacher"


@app.route('/final_result')
def final_result():
    return render_template("final.html")


@app.route("/navbar")

def navbar():
    return render_template("navbar.html")



@app.route('/signup')

def signup():
    return render_template('register.html')


def sign_up(Email_exists):
    return render_template("register.html", signal=Email_exists)


@app.route('/signup', methods=['GET','POST'])




def get_data_from_form():                                            
    reg_name=request.form['reg_name']                   
    reg_email=request.form['reg_email']
    reg_password=request.form['reg_password']
    reg_select=request.form['reg_select']
    result=check_email(reg_name,reg_email,reg_password,reg_select)
    return result
    
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
            return sign_up(Email_exists)
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
    
    return redirect(url_for('index'))




@app.route("/main")

def main():
    return render_template("main.html")


@app.route("/main",methods=['GET','POST'])

def data_from_teacher():
    student_name=request.form['st_name']
    subject_name=request.form['sub_name']
    answer_key=request.files['answer_key']
    student_email=request.form['st_email']
    student_answer=request.files['answer']
    total_marks=request.form['st_total_marks']
    answer=student_answer.read().decode()
    session['answer']=answer
    answer_keyy=answer_key.read().decode()
    session['answer_keyy']=answer_keyy


    result=PreProcessing.main(answer,answer_keyy,total_marks,student_name,student_email,subject_name)
    return result
    

@app.route('/logout')
def logout():
    # Clear the session and redirect to the login page
    session.clear()
    return redirect(url_for('index'))
 

app.run(debug=True)

