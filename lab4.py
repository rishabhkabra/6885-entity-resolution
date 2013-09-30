import json
import string
import re
from sklearn import svm

def edit_distance(first, second):
    """Find the Levenshtein distance between two strings."""
    if len(first) > len(second):
        first, second = second, first
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [[0] * second_length for x in range(first_length)]
    for i in range(first_length):
       distance_matrix[i][0] = i
    for j in range(second_length):
       distance_matrix[0][j]=j
    for i in xrange(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]

with open('locu_train_hard.json') as f:
    locu_train = json.loads(f.read())

with open('foursquare_train_hard.json') as f:
    fs_train = json.loads(f.read())

with open('matches_train_hard.csv') as f:
    matches_train = f.readlines()[1:] #locu_id, foursquare_id
    matches_train = [string.split(i, ',') for i in matches_train]
    matches_train = { (locu_id, fs_id) : 1 for (locu_id, fs_id) in matches_train}

wre = re.compile("\.[^\.]*\.[^\/]*")

def create_feature(l, f):
    feature = []
    feature.append(edit_distance(l['name'],f['name']))
    feature.append(1 if (l['postal_code']==f['postal_code'])  else 0)
    if wre.search(l['website']) and wre.search(f['website']):
        feature.append(1 if wre.search(l['website']).group() == wre.search(f['website']).group() else 0)
    else:
        feature.append(0)
    return feature

def create_feature_set(locu, fs, train = True):
    x = []
    y = []
    for l in locu:
        for f in fs:
            feature = create_feature(l,f)
            x.append(feature)
            if train:
                y.append(1 if (l['id'],f['id']) in matches_train else -1) 
    return (x, y) if train else x

x,y = create_feature_set(locu_train, fs_train)
clf = svm.SVC()
clf.fit(x,y)

with open('locu_test_hard.json') as f:
    locu_test = json.loads(f.read())

with open('foursquare_test_hard.json') as f:
    fs_test = json.loads(f.read())

x_test = create_features(locu_test, fs_test, False)
print clf.predict(x)
