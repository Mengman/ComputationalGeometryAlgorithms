# Rotating Calipers Algorithms (A.K.A Shamos Algorithm) use to find the diameter of contour
# This algorithm appeared in book (p178-p181) "Shamos, Franco P. Preparata, Michael Ian (1985). Computational Geometry An Introduction"
import cv2
import numpy as np
import math


def area(cnt, i, j, k):
    return cv2.contourArea(np.array([cnt[i], cnt[j], cnt[k]]))


def next_point(cnt, i):
    return (i + 1) % len(cnt)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_all_antipodal_pairs(cnt):
    p0 = 0
    p = len(cnt) - 1
    q = next_point(cnt, p)
    while area(cnt, p, next_point(cnt, p), next_point(cnt, q)) > area(cnt, p, next_point(cnt, p), q):
        q = next_point(cnt, q)

    q0 = q

    while q != p0:
        p = next_point(cnt, p)
        yield p, q

        while area(cnt, p, next_point(cnt, p), next_point(cnt, q)) > area(cnt, p, next_point(cnt, p), q):
            q = next_point(cnt, q)
            if (p, q) != (q0, p0):
                yield p, q

        if area(cnt, p, next_point(cnt, p), next_point(cnt, q)) == area(cnt, p, next_point(cnt, p), q):
            if (p, q) != (q0, len(cnt) - 1):
                yield p, next_point(cnt, q)
            else:
                break


def get_diameter(cnt):
    diameter = -1.0
    start = None
    end = None

    for i, j in get_all_antipodal_pairs(cnt):
        dist = distance(cnt[i][0], cnt[j][0])
        if dist > diameter:
            diameter = dist
            start = cnt[i][0]
            end = cnt[j][0]

    return diameter, start, end


if __name__ == '__main__':
    cnt = np.array(
        [
            [[10, 6]],
            [[9, 7]],
            [[8, 8]],
            [[7, 9]],
            [[7, 10]],
            [[7, 11]],
            [[7, 12]],
            [[7, 13]],
            [[8, 14]],
            [[8, 15]],
            [[8, 16]],
            [[9, 17]],
            [[10, 17]],
            [[11, 17]],
            [[11, 16]],
            [[12, 15]],
            [[13, 15]],
            [[14, 14]],
            [[15, 14]],
            [[15, 13]],
            [[16, 12]],
            [[16, 11]],
            [[15, 11]],
            [[14, 10]],
            [[14, 9]],
            [[13, 8]],
            [[13, 7]],
            [[12, 6]],
            [[11, 6]],
        ]
    )

    hull = cv2.convexHull(cnt, clockwise=False)

    diameter, start, end = get_diameter(hull)

    print(f"diameter: {diameter} start: {start} end: {end}")
