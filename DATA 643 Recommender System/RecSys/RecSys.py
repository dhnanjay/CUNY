from flask import Flask, request, render_template
import RecEngine
import loadData
#import userInput
import json
import urllib2
import re

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def my_form_post():

    movies = {}
    movies = loadData.loadMovies()
    prefs = {}
    prefs = loadData.loadRatings()
    movieArr=[]


    usrIp1 = request.form['1']
    if usrIp1 != " ":
        (user, movieid, rating, ts) = ('999', '1', usrIp1, '200912121604')
        movieArr.append([int(usrIp1), '1'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp2 = request.form['2']
    if int(usrIp2) > int(usrIp1):
        movieId="2"
    if usrIp2 != " ":
        (user, movieid, rating, ts) = ('999', '2', usrIp2, '200912121604')
        movieArr.append([int(usrIp2), '2'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp3 = request.form['3']
    if usrIp3 != " ":
        (user, movieid, rating, ts) = ('999', '3', usrIp3, '200912121604')
        movieArr.append([int(usrIp3), '3'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp4 = request.form['4']
    if usrIp4 != " ":
        (user, movieid, rating, ts) = ('999', '4', usrIp4, '200912121604')
        movieArr.append([int(usrIp4), '4'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp5 = request.form['5']
    if usrIp5 != " ":
        (user, movieid, rating, ts) = ('999', '5', usrIp5, '200912121604')
        movieArr.append([int(usrIp5), '5'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp6 = request.form['6']
    if usrIp6 != " ":
        (user, movieid, rating, ts) = ('999', '6', usrIp6, '200912121604')
        movieArr.append([int(usrIp6), '6'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp7 = request.form['7']
    if usrIp7 != " ":
        (user, movieid, rating, ts) = ('999', '7', usrIp7, '200912121604')
        movieArr.append([int(usrIp7), '7'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp8 = request.form['8']
    if usrIp8 != " ":
        (user, movieid, rating, ts) = ('999', '8', usrIp8, '200912121604')
        movieArr.append([int(usrIp8), '8'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp9 = request.form['9']
    if usrIp9 != " ":
        (user, movieid, rating, ts) = ('999', '9', usrIp9, '200912121604')
        movieArr.append([int(usrIp9), '9'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp10 = request.form['10']
    if usrIp10 != " ":
        (user, movieid, rating, ts) = ('999', '10', usrIp10, '200912121604')
        movieArr.append([int(usrIp10), '10'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp11 = request.form['11']
    if usrIp11 != " ":
        (user, movieid, rating, ts) = ('999', '11', usrIp11, '200912121604')
        movieArr.append([int(usrIp11), '11'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp12 = request.form['12']
    if usrIp12 != " ":
        (user, movieid, rating, ts) = ('999', '12', usrIp12, '200912121604')
        movieArr.append([int(usrIp12), '4'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp13 = request.form['13']
    if usrIp13 != " ":
        (user, movieid, rating, ts) = ('999', '13', usrIp13, '200912121604')
        movieArr.append([int(usrIp13), '13'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp14 = request.form['14']
    if usrIp14 != " ":
        (user, movieid, rating, ts) = ('999', '14', usrIp14, '200912121604')
        movieArr.append([int(usrIp14), '14'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp15 = request.form['15']
    if usrIp15 != " ":
        (user, movieid, rating, ts) = ('999', '15', usrIp15, '200912121604')
        movieArr.append([int(usrIp7), '15'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    usrIp16 = request.form['16']
    if usrIp16 != " ":
        (user, movieid, rating, ts) = ('999', '16', usrIp16, '200912121604')
        movieArr.append([int(usrIp16), '16'])
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    sorted(movieArr, key=lambda x: x[0], reverse=True)
# Get highest rate3d movie
    tRtM = movieArr[0]
    toprtMov = tRtM[1]
    print(toprtMov)

# Get 3 users with most similarinterest User User CF
    topMatch = RecEngine.topMatches(prefs, '999', n=3)

    recommendedMovies = RecEngine.getRecommendations(prefs, '999', n=5)
# API
    url = "https://api.themoviedb.org/3/search/movie?api_key=e8a3a16cda5baf6f1acd31fc04484897&query="
    iUrl = "https://image.tmdb.org/t/p/w300"


    movie1 = recommendedMovies[0]
    text1 = movie1[1]
    title1 = text1[:-7]
    title1 = title1.replace(" ", "+")
    url1 = url+title1
    mData1 = json.load(urllib2.urlopen(url1))
    date1 = mData1["results"][0]["release_date"]
    ov1 = mData1["results"][0]["overview"]
    image1 = mData1["results"][0]["poster_path"]
    image1 = iUrl+image1


    movie2 = recommendedMovies[1]
    text2 = movie2[1]
    title2 = text2[:-7]
    title2 = title2.replace(" ", "+")
    url2 = url + title2
    mData2 = json.load(urllib2.urlopen(url2))
    date2 = mData2["results"][0]["release_date"]
    ov2 = mData2["results"][0]["overview"]
    image2 = mData2["results"][0]["poster_path"]
    image2 = iUrl + image2

    movie3 = recommendedMovies[2]
    text3 = movie3[1]
    title3 = text3[:-7]
    title3 = title3.replace(" ", "+")
    url3 = url + title3
    mData3 = json.load(urllib2.urlopen(url3))
    date3 = mData3["results"][0]["release_date"]
    ov3 = mData3["results"][0]["overview"]
    image3 = mData3["results"][0]["poster_path"]
    image3 = iUrl + image3

    movie4 = recommendedMovies[3]
    text4 = movie4[1]
    title4 = text4[:-7]
    title4 = title4.replace(" ", "+")
    url4 = url + title4
    mData4 = json.load(urllib2.urlopen(url4))
    date4 = mData4["results"][0]["release_date"]
    ov4 = mData4["results"][0]["overview"]
    image4 = mData4["results"][0]["poster_path"]
    image4 = iUrl + image4

    movie5 = recommendedMovies[4]
    text5 = movie5[1]
    title5 = text5[:-7]
    title5 = title5.replace(" ", "+")
    url5 = url + title5
    mData5 = json.load(urllib2.urlopen(url5))
    date5 = mData5["results"][0]["release_date"]
    ov5 = mData5["results"][0]["overview"]
    image5 = mData5["results"][0]["poster_path"]
    image5 = iUrl + image5

# Transform dataset to get Rec. based Item Item CF

    moviesDS = RecEngine.transformPrefs(prefs)
    itemRec = moviesDS
    try:
        itemRec = RecEngine.topMatches(moviesDS, '20', n=5)
    except Exception,e:
        print(e)

    return render_template("index.html", title1=text1[:-7], date1=date1, Rating1=movie1[0], overview1=ov1,p_url1=image1,
                           title2=text2[:-7], date2=date2, Rating2=movie2[0], overview2=ov2, p_url2=image2,
                           title3=text3[:-7], date3=date3, Rating3=movie3[0], overview3=ov3, p_url3=image3,
                           title4=text4[:-7], date4=date4, Rating4=movie4[0], overview4=ov4, p_url4=image4, itemRec = itemRec,
                           title5=text5[:-7], date5=date5, Rating5=movie5[0], overview5=ov5, p_url5=image5, topMatch = topMatch)



if __name__ == '__main__':
    app.run()
