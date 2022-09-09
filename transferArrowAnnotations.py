import json
import numpy as np

datasetFilename = '/Users/murtuza/Desktop/data.json'
arrowSplitDatasetFileName = '/Users/murtuza/Desktop/arrows.json'
arrowAnnotationsFileName = '/Users/murtuza/Desktop/arrowsAnnotations.json'

data = json.load(open(datasetFilename, 'r'))
arrows = json.load(open(arrowSplitDatasetFileName, 'r'))
arrowsAnnotations = json.load(open(arrowAnnotationsFileName, 'r'))

arrowData = data['Arrow']
print("#arrows in data = {}".format(len(arrowData)))


def comparePoints(points1, points2):
    if len(points1) != len(points2):
        return False
    for i in range(len(points1)):
        p_diff = np.array(points1[i]) - np.array(points2[i])
        if np.linalg.norm(p_diff) > 2:
            return False
    return True


keysMatched = []
for key, points in arrows.items():
    for i in range(len(arrowData)):
        pointsData = arrowData[i]['points']
        if comparePoints(points, pointsData) and arrowData[i]['arrowHeadStartIndex'] == -1:
            arrowData[i]['arrowHeadStartIndex'] = arrowsAnnotations[key]
            keysMatched.append(key)
            break

data['Arrow'] = arrowData
# json.dump(data, open('/Users/murtuza/Desktop/dataTrue.json', 'w'))

count = 0
for d in arrowData:
    if d['arrowHeadStartIndex'] == -1:
        count += 1
print("not annotated samples - {}".format(count))

print("#annotation transferred = {}".format(len(keysMatched)))
print("------no match found--------")
print(len(set(arrows.keys()) - set(keysMatched)))
print(set(arrows.keys()) - set(keysMatched))
