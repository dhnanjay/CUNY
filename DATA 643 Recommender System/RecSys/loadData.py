
movies = {}
prefs = {}
def loadMovies():
    # Load Movie titles and Ids /home/dhananjay/PycharmProjects/RecSys/static/u.item  /home/dhananjay/PycharmProjects/RecSys/static/u.data
    for line in open('/u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title

    return movies

def loadRatings():
    # Load User Ratings

    for line in open('static/u.data'):
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)

    return prefs

