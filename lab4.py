import json
import string
import re
from sklearn import svm

def edit_distance(first, second):
    """Find the Levenshtein distance between two strings.
    Original method written by Stavros Korokithakis."""
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
    lines = f.readlines()[1:] #locu_id, foursquare_id
    matches_train = [string.split(line.strip(), ',') for line in lines]
    matches_train = {(locu_id,fs_id):1 for (locu_id, fs_id) in matches_train}

wre = re.compile("\.[^\.]*\.[^\/]*")

def create_feature(l, f):
    feature = []
    #check names
    edist = edit_distance(l['name'],f['name'])
    edist /= max(1, len(l['name']), len(f['name']))
    feature.append(edist)
    #check latitude and longitude
    if (not(l['latitude'] and f['latitude'])):
      feature.append(0)
    else:
      lat = abs(l['latitude'] -  f['latitude'])
      feature.append(1 if lat < 0.1 else -1)
    if (not(l['longitude'] and f['longitude'])):
      feature.append(0)
    else:
      lng = abs(l['longitude'] -  f['longitude'])
      feature.append(1 if lng < 0.1 else -1)
    #check postal codes
    if (not(l['postal_code'] and f['postal_code'])):
      feature.append(0)
    else:
      feature.append(1 if (l['postal_code']==f['postal_code'])  else -1)
    #check websites
    lweb = wre.search(l['website'])
    fweb = wre.search(f['website'])
    if (lweb is not None and fweb is not None):
      feature.append(1 if lweb.group() == fweb.group() else -1)
    else:
      feature.append(0)
    #check phones
    if (not(l['phone'] and f['phone'])):
        feature.append(0)
    elif re.sub(r"\D", "", l['phone']) == re.sub(r"\D", "", f['phone']):
        feature.append(1)
    else:
        feature.append(-1)
    # check addresses
    # TODO: replace address abbreviations. Think if first part of it is exactly same.                                                                                             
    laddr = re.findall(re.compile(r'\b\w[\D]'), l['street_address']).lowercase()
    faddr = re.findall(re.compile(r'\b\w[\D]'), f['street_address']).lowercase()
    feature.append(1 - (edit_distance(laddr, faddr) / max(len(l['street_address']), len(f['street_address']), 1)))
    return feature

def create_feature_set(locu, fs, matches = {}):
    x = []
    y = []
    for l in locu:
        for f in fs:
            feature = create_feature(l,f)
            if feature:
              x.append(feature)
	    if l['id'] in matches and matches[l['id']] == f['id']:
            	y.append(1)
            else:
                y.append(-1)
    return (x, y)

print "making feature vectors for training data..."
x,y = create_feature_set(locu_train, fs_train, matches_train)
clf = svm.SVC()
print "fitting training data ..."
clf.fit(x,y)

with open('locu_test_hard.json') as f:
    locu_test = json.loads(f.read())

with open('foursquare_test_hard.json') as f:
    fs_test = json.loads(f.read())

print "making feature vectors for test data..."
x_test = create_feature_set(locu_test, fs_test)[0]
print "predicting testing data..."
y_test = clf.predict(x_test)

matches_file = open('matches_test.csv', 'w')
yindex = 0
for l in locu_test:
    for f in fs_test:
        if y_test[yindex] == 1:
            output = '{0},{1}\n'.format(l['id'],f['id'])
            matches_file.write(output)
        yindex += 1
matches_file.close()
