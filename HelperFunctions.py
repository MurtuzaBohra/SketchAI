# create csv files from the json exported after reading the napkin dataset.
import json
import csv
import re
import numpy as np


def readNapkinStrokesToCSV(pathNapkinData, outputPath):
    # pathNapkinData = '/Users/murtuza/Desktop/ShapeRecognition/deep-stroke/dataset/NapkinData'
    file = open(pathNapkinData, 'r')
    data = json.load(file)
    print(data.keys())
    for k, value in data.items():
        k = re.sub(r"-", " ", k)
        shapeName = '_'.join(k.split())
        for idx, exampleShape in enumerate(value):
            f = open("{}/TestCsv/{}-{}.csv".format(outputPath, shapeName, idx), "w", newline="")
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
    return svgPath


def readJsonContentTestData(filename):
    jsonContent = json.load(open(filename, 'r'))
    inputStroke = []
    if type(jsonContent) == list:
        for point in jsonContent[0]['points']:
            inputStroke.append(np.array([point['x'], point['y']]))
            # if include_finger
            # inputStroke.append(np.array([point['x'], point['y'], 0]))
        inputStroke = np.array(inputStroke)
    else:
        try:
            inputStroke = np.array(jsonContent['input']['context']['graph']['nodes'][0]['curves'][0]).reshape((-1, 3))[
                          :, :2]
        except:
            inputStroke = np.array(jsonContent['body']['input']['context']['graph']['nodes'][0]['curves'][0]).reshape(
                (-1, 3))[
                          :, :2]

        # if include_finger
        # inputStroke = np.array(jsonContent['input']['context']['graph']['nodes'][0]['curves'][0]).reshape((-1, 3))[:, :2]
        # inputStroke = np.concatenate((inputStroke, np.zeros((inputStroke.shape[0],1))), axis=1)
    inputStroke -= np.min(inputStroke, axis=0)#inputStroke[0]
    return inputStroke


# -------------------------------------------------------------
# -------------------------------------------------------------
class LineEquation:
    def __init__(self, a=None, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c

    def computeLineFromPoints(self, p0, p1):
        self.a = p1[1] - p0[1]
        self.b = p0[0] - p1[0]
        self.c = p0[0] * (-self.a) - p0[1] * self.b

    def distanceFromPoint(self, p):
        # return the signed distance of a point from the line
        d = (self.a * p[0] + self.b * p[1] + self.c) / np.sqrt(self.a * self.a + self.b * self.b)
        return d

    def getSlope(self):
        return np.arctan(-(self.a / self.b))

    def findReflectionOfPoint(self, p):
        t = (-2 * ((self.a * p[0]) + (self.b * p[1]) + self.c)) / (np.square(self.a) + np.square(self.b))
        x = self.a * t + p[0]
        y = self.b * t + p[1]
        return [x, y]


def flipOrientation(autoSuggestionPoints):
    line = LineEquation()
    line.computeLineFromPoints(autoSuggestionPoints[0], autoSuggestionPoints[-1])
    autoSuggestionPoints_flipped = findReflection(autoSuggestionPoints, line)
    autoSuggestionPoints_flipped = np.array(autoSuggestionPoints_flipped) - np.min(autoSuggestionPoints_flipped, axis=0)
    return autoSuggestionPoints_flipped


def distanceBtwPoints(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))


def fitOrientation(userStrokePoints, autoSuggestionPoints, line):
    nPoints = userStrokePoints.shape[0]
    start = int(np.floor(nPoints * 0.25))
    end = int(np.ceil(nPoints * 0.75))
    middlePoints = userStrokePoints[start:end, :]
    user_distance = 0
    for i in range(middlePoints.shape[0]):
        user_distance += line.distanceFromPoint(middlePoints[i])

    distance_start_start = distanceBtwPoints(userStrokePoints[0], autoSuggestionPoints[0])
    distance_start_end = distanceBtwPoints(userStrokePoints[0], autoSuggestionPoints[-1])
    autoLine = LineEquation()
    autoLine.computeLineFromPoints(autoSuggestionPoints[0], autoSuggestionPoints[-1])
    if distance_start_start > distance_start_end:
        autoLine.computeLineFromPoints(autoSuggestionPoints[-1], autoSuggestionPoints[0])

    nPoints = autoSuggestionPoints.shape[0]
    start = int(np.floor(nPoints * 0.25))
    end = int(np.ceil(nPoints * 0.75))
    middlePoints = autoSuggestionPoints[start:end, :]
    auto_distance = 0
    for i in range(middlePoints.shape[0]):
        auto_distance += autoLine.distanceFromPoint(middlePoints[i])

    if auto_distance * user_distance < 0:
        return flipOrientation(autoSuggestionPoints)
    else:
        return autoSuggestionPoints


def getCenter(points):
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def findReflection(points, line):
    reflectedPoints = []
    for i in range(points.shape[0]):
        x, y = line.findReflectionOfPoint(points[i])
        reflectedPoints.append([x, y])
    reflectedPoints = np.array(reflectedPoints)
    return reflectedPoints


def getAutoSuggestBracket(userStrokePoints, templateBracket):
    line = LineEquation()
    line.computeLineFromPoints(userStrokePoints[0], userStrokePoints[-1])
    slopeRadian = line.getSlope()

    xCenter, yCenter = getCenter(templateBracket)

    autoSuggestion = []
    if abs(slopeRadian) > 0.3:
        for i in range(templateBracket.shape[0]):
            x = np.cos(slopeRadian) * (templateBracket[i, 0] - xCenter) - np.sin(slopeRadian) * (
                    templateBracket[i, 1] - yCenter) + xCenter
            y = np.sin(slopeRadian) * (templateBracket[i, 0] - xCenter) + np.cos(slopeRadian) * (
                    templateBracket[i, 1] - yCenter) + yCenter
            autoSuggestion.append([x, y])
        autoSuggestion = np.array(autoSuggestion) - np.min(autoSuggestion, axis=0)
    else:
        autoSuggestion = templateBracket

    autoSuggestion = fitOrientation(userStrokePoints, autoSuggestion, line)
    autoSuggestion = np.array(autoSuggestion) - np.min(autoSuggestion, axis=0)
    return np.array(autoSuggestion)


# def getPerpendicularLine(line, point):
#     m = line.b / line.a
#     c = point[1] - m * point[0]
#     a = -c / (point[0] + m * point[1])
#     b = m * a
#     return a, b, c


def shiftOrigin(points):
    # shift origin from topLeft to bottomRight or vice versa.
    _, y = np.max(points, axis=0)
    x, _ = np.min(points, axis=0)
    points = points - np.array([x, y])
    points[:, 1] = -1 * points[:, 1]
    return points


# -------------------------------------------------------------
# -------------------------------------------------------------

def writeStringToFile(filename, strng):
    # file = open('/Users/murtuza/Desktop/test.svg', 'w')
    file = open(filename, 'w')
    file.write(strng)
    file.close()


if __name__ == "__main__":
    readNapkinStrokesToCSV('/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/NapkinTestStrokes.json', '/Users/murtuza/Desktop/SketchAI/dataset/NapkinData/')

    # userStrokePoints = readJsonContentTestData('/Users/murtuza/Desktop/brackets/w_curly_braces.json')
    # templateBracket = readJsonContentTestData('/Users/murtuza/Desktop/brackets/squareBracketTemplate.json')
    # print([list(item) for item in templateBracket.round(2)])
    # writeStringToFile('/Users/murtuza/Desktop/squareBracketTemplate.svg', convert_curve_points_to_svg(templateBracket))
    # # writeStringToFile('/Users/murtuza/Desktop/brackets/square_bracket.svg',
    # #                   convert_curve_points_to_svg(userStrokePoints))
    #
    # # userStrokePoints, templateBracket = shiftOrigin(userStrokePoints), shiftOrigin(templateBracket)
    # autoSuggestionBracket = getAutoSuggestBracket(userStrokePoints, templateBracket)
    # # autoSuggestionBracket = shiftOrigin(autoSuggestionBracket)
    # writeStringToFile('/Users/murtuza/Desktop/test.svg', convert_curve_points_to_svg(autoSuggestionBracket))

