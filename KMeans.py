# Implementation of K-Means Clustering
# For future pre-flop implementations, as well as hand bucketing based on equities
# Not currently used
import json
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import CardUtils

class KMeans:

    #Values is a Dict
    def __init__(self, points):
        self.points = points

    def getKMeans(self, k):
        means, distortion = kmeans(self.points, k)
        return means

if __name__ == "__main__":
    with open('preFlopEquitiesSqRootLong.json') as file:
        data = json.load(file)
    equityPoints = list(data.values())
    preEquityKMeans = KMeans(equityPoints)
    means = preEquityKMeans.getKMeans(20)
    buckets = {mean: [] for mean in means}
    for key, value in data.items():
        smallestDiff = 100
        closestMean = -1
        for mean in means:
            if abs(mean - value) < smallestDiff:
                closestMean = mean
                smallestDiff = abs(mean - value)
        buckets[closestMean] += [key]
    sortedKeys = sorted(buckets.keys())
    for key in sortedKeys:
        print(key, CardUtils.listNumStringsToListHands(buckets[key]), '\n')
         
