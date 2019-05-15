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
    list2 = []
    for index, row in df_emp.iterrows():
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            list2.append([i, 6 - index, row[i]])

    return render_template("schedule.html",emData=list2)




if __name__ == '__main__':
    app.run()
