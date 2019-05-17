from flask import Flask,render_template,request, redirect, url_for,session
#from forms import TestForms
import sqlalchemy
import psycopg2
import random
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pyschedule
from pyschedule import Scenario, solvers, plotters, alt
from main import *

app = Flask(__name__)

app.secret_key = 'You Will Never Guess'

@app.route('/index')
def hello_world():

  #  gData=[[0, 1, 18], [0, 0, 19], [0, 2, 8], [0, 3, 24], [0, 4, 67], [1, 0, 92], [1, 1, 58], [1, 2, 78], [1, 3, 117], [1, 4, 48], [2, 0, 35], [2, 1, 15], [2, 2, 123], [2, 3, 64], [2, 4, 52], [3, 0, 72], [3, 1, 132], [3, 2, 114], [3, 3, 19], [3, 4, 16], [4, 0, 38], [4, 1, 5], [4, 2, 8], [4, 3, 117], [4, 4, 115], [5, 0, 88], [5, 1, 32], [5, 2, 12], [5, 3, 6], [5, 4, 120], [6, 0, 13], [6, 1, 44], [6, 2, 88], [6, 3, 98], [6, 4, 96], [7, 0, 31], [7, 1, 1], [7, 2, 82], [7, 3, 32], [7, 4, 30], [8, 0, 85], [8, 1, 97], [8, 2, 123], [8, 3, 64], [8, 4, 84], [9, 0, 47], [9, 1, 114], [9, 2, 31], [9, 3, 48], [9, 4, 91]]
    df_traffic = get_traffic()
    list1 = []
    for index, row in df_traffic.iterrows():
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            list1.append([i, 6 - index, row[i]])
    print("traffic")

    try:
        #df2=get_schedule()
        print("Hi")
    except:
        print("Error catch")

    return render_template("index.html",trData=list1)

@app.route('/index', methods=['POST'])
def my_form_post():

    session['minSeq'] = request.form['min_seq']
    session['maxSeq'] = request.form['max_seq']
    session['minWork'] = request.form['min_work']
    session['maxWork'] = request.form['max_work']
    session['TER'] = request.form['TER']

    return redirect(url_for('schedule'))


@app.route('/schedule')
def schedule():
    #print("m:",session['maxSeq'])
    oneEmp = float(session['TER'])
    min_seq = int(session['minSeq'])
    max_seq = int(session['maxSeq'])
    min_work = int(session['minWork'])
    max_work = int(session['maxWork'])
    #print(oneEmp,session['minSeq'],session['maxSeq'])
    df_sol, df_emp = get_schedule(oneEmp=oneEmp,n_hours=12,min_seq=min_seq,max_seq=max_seq,
                                  min_work=min_work,max_work=max_work)
    #df_sol, df_emp = get_schedule()
    manHours = df_emp.sum().sum()
    print("manhours",manHours)

    ## Employee Needed Grid
    list2 = []
    for index, row in df_emp.iterrows():
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            list2.append([i, 6 - index, row[i]])
    print("list2")
    ## Schedule
    df_sol2 = df_sol.pivot_table(values='Available', index=['Day', 'Title'], columns='TimeSlot', fill_value=0,
                                   aggfunc='sum')
    print("df_sol2")
    iList=[]
    for i in df_sol2.index.tolist():
        if i[0] == 0.0:
            iList.append('Sunday ' + i[1])
        elif i[0] == 1.0:
            iList.append('Monday ' + i[1])
        elif i[0] == 2.0:
            iList.append('Tuesday ' + i[1])
        elif i[0] == 3.0:
            iList.append('Wednesday ' + i[1])
        elif i[0] == 4.0:
            iList.append('Thursday ' + i[1])
        elif i[0] == 5.0:
            iList.append('Friday ' + i[1])
        elif i[0] == 6.0:
            iList.append('Saturday ' + i[1])

    print("iList")

    iList2 = []
    for index, row in df_sol2.iterrows():
        for i in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            iList2.append([i, 6 - index[0], row[i]])

    print("iList2")

    outputList = []
    for index, row in df_sol2.iterrows():
        hours = 0
        Day=""
        for j in list(row):
            hours = hours + int(j)
        if index[0] == 0.0:
            Day = 'Sunday'
        elif index[0] == 1.0:
            Day = 'Monday'
        elif index[0] == 2.0:
            Day = 'Tuesday'
        elif index[0] == 3.0:
            Day = 'Wednesday'
        elif index[0] == 4.0:
            Day = 'Thursday'
        elif index[0] == 5.0:
            Day = 'Friday'
        elif index[0] == 6.0:
            Day = 'Saturday'
        outputList.append([Day, index[1],str(row[9]),str(row[10]),str(row[11]),str(row[12]),
                           str(row[13]),str(row[14]),str(row[15]),str(row[16]),str(row[17]),
                           str(row[18]),str(row[19]),str(row[20]),str(hours)])
    try:
        return render_template("schedule.html", emData=list2, idHours=manHours,
                           fHours=df_sol.Available.sum(),
                           smHours=str(df_sol.loc[df_sol['Title'] == 'SM'].Available.sum()),
                           asmHours=str(df_sol.loc[df_sol['Title'] == 'ASM'].Available.sum()),
                           t1Hours=str(df_sol.loc[df_sol['Title'] == 'TEMP1'].Available.sum()),
                           t2Hours=str(df_sol.loc[df_sol['Title'] == 'TEMP2'].Available.sum()),
                           t3Hours=str(df_sol.loc[df_sol['Title'] == 'TEMP3'].Available.sum()),
                           dList=outputList)
    except:
        print("Error")




if __name__ == '__main__':
    app.run()
