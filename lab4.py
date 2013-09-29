import json
import string
from sklearn import svm

with open('locu_train_hard.json') as f:
    locu_train = json.loads(f.read())

with open('foursquare_train_hard.json') as f:
    fs_train = json.loads(f.read())

with open('matches_train_hard.csv') as f:
    matches_train = f.readlines()[1:] #locu_id, foursquare_id
    matches_train = [string.split(i, ',') for i in matches_train]
    matches_train = {locu_id: fs_id for (locu_id, fs_id) in matches_train}

def create_feature(l, f):
    feature = [l,f] #change this
    return feature

def create_feature_set(locu, fs):
    x = []
    for l in locu_train:
        for f in fs_train:
            feature = create_feature(l,f)
            x.append(feature)
    return x

x = create_feature_set(locu_train, fs_train)
y = []
for l in locu_train:
    if l["id"] in matches_train:
        y.append(1)
    else:
        y.append(-1)
clf = svm.SVC()
clf.fit(x,y)

with open('locu_test_hard.json') as f:
    locu_test = json.loads(f.read())

with open('foursquare_test_hard.json') as f:
    fs_test = json.loads(f.read())

x = create_features(locu_test, fs_test)
clf.predict(x)
