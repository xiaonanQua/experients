
'''
1 找到最远的两个点作为聚类中心
2 求所有点到这两个点的距离，并取这两个值中的最小值，找到最大值设为第三个聚类中心
3 求所有的点到这三个聚类中心的距离，并归入该聚类
'''





import numpy as np

matrix = np.array([[0,0],[3,8],[2,2],[1,1],[5,3],[4,8],[6,3],[5,4],[6,4],[7,5]])
print(matrix)

distance = np.arange(100)
distance = distance.reshape((10,10))

for i in range(10):
    for j in range(10):
        distance[i][j] = 0

print(distance)

max = 0

a,b = 0,0

for i in range(10):
    for j in range(i+1,10):
        distance[i][j] = (matrix[i][0] - matrix[j][0]) ** 2 + (matrix[i][1] - matrix[j][1]) ** 2
        distance[j][i] = distance[i][j]
        if distance[i][j] > max:
            max = distance[i][j]
            a,b = i,j
print(distance)


print(a,b,max)

# 已找到两个聚类





max_distance1 = 0
max_i = 0


for i in range(10):
    if max_distance1<min(distance[i][a],distance[i][b]):
        max_distance1 = min(distance[i][a],distance[i][b])
        max_i = i
c = max_i
# i 为第三个聚类中心 0 , 5 ,6
# 聚类为a，b，c

print(max_i)


cluster_a = set()
cluster_a.add(a)
cluster_b = set()
cluster_b.add(b)
cluster_c = set()
cluster_c.add(c)

(1)




for i in range(10):
    if i != a and i != b and i != c:
        min_distance1 = 1000000
        min_x = 0
        for k in (a,b,c):
            if min_distance1 > distance[i][k]:
                min_distance1 = distance[i][k]
                min_x = k
        if min_x == b:
            cluster_b.add(i)
        if min_x == a:
            cluster_a.add(i)

        if min_x == c:
            cluster_c.add(i)

print(cluster_a,cluster_b,cluster_c)
