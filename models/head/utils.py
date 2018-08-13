import numpy as np
import math
def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA())**0.5
    return np.matrix(ED)
def point_distances(src_points, gt_points):
    """
    determine distances between src_points and gt_points
    """
    distances = EuclideanDistances(np.matrix(src_points), np.matrix(gt_points))
    return np.array(distances)
