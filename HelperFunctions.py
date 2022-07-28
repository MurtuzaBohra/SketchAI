# create csv files from the json exported after reading the napkin dataset.
import json
import csv
import re

def readNapkinStrokesToCSV(pathNapkinData):
    # pathNapkinData = '/Users/murtuza/Desktop/ShapeRecognition/deep-stroke/dataset/NapkinData'
    file = open("{}/NapkinStrokes.json".format(pathNapkinData), 'r')
    data = json.load(file)
    print(data.keys())
    for k,value in data.items():
        k = re.sub(r"-", " ", k)
        shapeName = '_'.join(k.split())
        for idx, exampleShape in enumerate(value):
            f = open("{}/csv/{}-{}.csv".format(pathNapkinData,shapeName, idx), "w", newline="")
            writer = csv.writer(f)
            writer.writerows([["stroke_id", "x", "y", "time", "is_writing"]])
            writer.writerows(exampleShape)
            f.close()


def convert_curve_points_to_svg(points):
    svgPath = "<svg viewBox=\"0 0 300 300\" position=\"absolute\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M "
    for i in range(points.shape[0]):
        svgPath += str(points[i, 0].round()) + " " + str(points[i, 1].round()) + ' L '
    svgPath = svgPath[:-3]
    svgPath += '\" fill=\"#ffff\" stroke=\"black\"></path></svg>'
    print(svgPath)