import numpy as np

# import matplotlib.pyplot as plt

file = open("Tests/polys_1.txt", "r")
m = []
lines = file.readlines()
h = len(lines)
w = 0
for l in lines:
    for c in l:
        if c == "0":
            m.append(0)
        if c == "1":
            m.append(1)
m = np.array(m).reshape(h, len(m) // h)

clusters = []
c = m.astype(int)


class Point:
    def Point(self):
        x = 0
        y = 0


def get_value_at(m, x, y):
    if x < 0 or y < 0:
        return 0
    if x >= m.shape[0] or y >= m.shape[1]:
        return 0
    return np.maximum(0, m[x, y])


def get_neighbours(m, x, y):
    n = list()
    # n.append(get_value_at(m, x + 1, y + 1))
    n.append(get_value_at(m, x + 0, y + 1))
    # n.append(get_value_at(m, x + -1, y + 1))
    n.append(get_value_at(m, x + 1, y + 0))
    n.append(get_value_at(m, x + -1, y + 0))
    # n.append(get_value_at(m, x + 1, y + -1))
    n.append(get_value_at(m, x + 0, y + -1))
    # n.append(get_value_at(m, x + -1, y + -1))
    return n


def is_edge_pixel(m, x, y):
    neighbours = get_neighbours(m, x, y)
    if 0 in neighbours and m[x, y] > 0:
        return True


def search_contour(block, fill):
    block_cp = np.copy(block)
    it = np.nditer(block, flags=['multi_index'])
    while not it.finished:
        x = it.multi_index[0]
        y = it.multi_index[1]
        if is_edge_pixel(block, x, y):
            block_cp[x, y] = fill
            # found = True
        it.iternext()
    return block_cp


def get_score(chunk, r, c):
    nc = [-1, 0, 1, -1, 1, -1, 0, 1]
    nr = [-1, -1, -1, 0, 0, 1, 1, 1]
    s = 0
    for p in zip(nr, nc):
        sr = r + p[0]
        sc = c + p[1]
        if sr >= 0 and sr <= 2 and sc >= 0 and sc <= 2:
            s += chunk[sr, sc]
    return s


def next_poly_point(block, r, c, poly_points):
    nc = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    nr = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    chunk = list()
    for p in zip(nr, nc):
        chunk.append(get_value_at(block, r + p[0], c + p[1]))
    chunk = np.array(chunk).reshape((3, 3))

    chunk[1, 1] = 0
    scores = np.zeros_like(chunk)
    # evulate score for each element of chunk array
    it = np.nditer(chunk, flags=['multi_index'])
    while not it.finished:
        ir = it.multi_index[0]
        ic = it.multi_index[1]
        if chunk[ir, ic] > 0 and [ir-1+r,ic-1+c] not in poly_points:
            scores[ir, ic] = get_score(chunk, ir, ic)
        else:
            scores[ir, ic] = 1000000
        it.iternext()
    p = list(np.unravel_index( np.argmin(scores), scores.shape))
    p[0] = p[0] - 1 + r
    p[1] = p[1] - 1 + c
    return p

def buid_polygon(block, num):
    # scan
    # scan
    ones = block <= num
    zeros = block > num
    block[ones] = 1
    block[zeros] = 0
    poly = list()
    seed = [0, 0]
    while True:
        p = next_poly_point(block, seed[0], seed[1], poly)
        if p[0] == p[1] == -1:
            break
        poly.append(p)
        seed = p
    return poly


fill = -1
while True:
    c2 = search_contour(c, fill)
    if np.array_equal(c, c2):
        break
    fill -= 1
    c = c2
print(c)
poly = buid_polygon(c, -1)
print(poly)

# plt.axes()
#
# circle = plt.Circle((0, 0), radius=0.75, fc='y')
# plt.gca().add_patch(circle)
#
# plt.axis('scaled')
# plt.show()
