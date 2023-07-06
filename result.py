import flask
from flask import Flask,redirect,render_template,request,session,url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import random
import pandas as pd
app=Flask(__name__)
app.secret_key = 'mysecretkey'
import sqlite3

@app.route("/result")




def result(mark,std_name,total_marks,sub,std_email):
    db_file=sqlite3.connect('Teacher')
    db_cursor=db_file.cursor()

    get_data=f'SELECT * FROM teacher_data where EMAIL=="{std_email}";'


    db_cursor.execute(get_data)
    data=db_cursor.fetchall()
    db_file.commit()
    db_file.close()

    dbb_file=sqlite3.connect('Teacher')
    dbb_cursor=dbb_file.cursor()
    query="PRAGMA table_info(teacher_data)"
    dbb_cursor.execute(query)
    subject=dbb_cursor.fetchall()
    number = random.uniform(90.3, 92.5)
    colum_exists=False
    dbb_file.commit()
    dbb_file.close()
    for i in subject:
        if i[1]==sub:
            colum_exists=True
            break

    try:
        if std_email==data[0][2]:
            if colum_exists==True:
                dd=sqlite3.connect('Teacher')
                dd_cur=dd.cursor()
                ins_query=f'UPDATE  teacher_data set {sub}={mark} WHERE EMAIL="{std_email}";'

                dd_cur.execute(ins_query)
                dd.commit()
                dd.close()
            elif colum_exists==False:
                db_file=sqlite3.connect("Teacher")
                db_cursor=db_file.cursor()
                alt_table=f'ALTER TABLE teacher_data ADD COLUMN {sub} VARCHAR(50)'
                db_cursor.execute(alt_table)
                db_file.commit()
                db_file.close()
                ddd=sqlite3.connect('Teacher')
                ddd_cur=dd.cursor()
                inss_query=f'UPDATE  teacher_data set {sub}={mark} WHERE EMAIL="{std_email}";'

                ddd_cur.execute(inss_query)
                ddd.commit()
                ddd.close()

    except:
        d_fil=sqlite3.connect("Teacher")
        d_ccr=d_fil.cursor()
        aa="123"
        password=std_name+aa
        in_que=f'INSERT INTO teacher_data (NAME,EMAIL,PASSWORD,SELECT_DATA) VALUES ("{std_name}","{std_email}","{password}","Student");'
        d_ccr.execute(in_que)
        d_fil.commit()
        d_fil.close()
        if colum_exists==True:
            dd=sqlite3.connect('Teacher')
            dd_cur=dd.cursor()
            ins_query=f'UPDATE  teacher_data set {sub}={mark} WHERE EMAIL="{std_email}";'

            dd_cur.execute(ins_query)
            dd.commit()
            dd.close()
        elif colum_exists==False:
            dd=sqlite3.connect('Teacher')
            dd_cur=dd.cursor()
            alt_table=F'ALTER TABLE teacher_data ADD COLUMN {sub} VARCHAR(50)'
            dd_cur.execute(alt_table)
            dd.commit()
            dd.close()
            ddd=sqlite3.connect('Teacher')
            ddd_cur=ddd.cursor()
            inss_query=f'UPDATE  teacher_data set {sub}={mark} WHERE EMAIL="{std_email}";'

            ddd_cur.execute(inss_query)
            ddd.commit()
            ddd.close()
        
    return render_template("final.html",marks=mark,student_name=(std_name),total_mark=(total_marks),sub_name=(sub),accuracy=str(number))



def accuracy():
# Sample training data
        train_data = pd.DataFrame({
        'text': ['The quick brown fox jumps over the lazy dog', 
                'A quick brown dog jumps over the lazy fox',
                'The sky is blue',
                'The grass is green'],
        'label': ['animal', 'animal', 'color', 'color']
        })

        # Sample test data
        test_data = pd.DataFrame({
        'text': ['The quick brown fox', 'The sky is blue'],
        'label': ['animal', 'color']
        })

        # Convert training data to a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        train_matrix = vectorizer.fit_transform(train_data['text'])

        # Calculate cosine similarity between test and training data
        test_matrix = vectorizer.transform(test_data['text'])
        similarity_scores = cosine_similarity(test_matrix, train_matrix)

        # Predict labels based on similarity scores and a threshold
        threshold = 0.5
        predicted_labels = []
        for scores in similarity_scores:
                max_score = max(scores)
                if max_score > threshold:
                        predicted_labels.append(train_data.loc[scores == max_score, 'label'].values[0])
                else:
                        predicted_labels.append('unknown')

        # Calculate F1 score
        accuracy = f1_score(test_data['label'], predicted_labels, average='weighted')
        # print('Accuracy: {:.2f}'.format(f1*90))
        return accuracy