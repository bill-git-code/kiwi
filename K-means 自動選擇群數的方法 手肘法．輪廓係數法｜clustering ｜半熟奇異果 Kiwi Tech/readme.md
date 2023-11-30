# K-means 自動選擇群數的方法: 手肘法．輪廓係數法｜clustering ｜半熟奇異果 Kiwi Tech
kmeans; elbow method; Silhouette Coefficient; 集群分析, 分群; algorithm; machine learning; AI

![KiwiTech_title_長條03](https://github.com/bill-git-code/kiwi/assets/63846041/2618c4f7-a5d9-4edf-90ef-3ee188860471)


---

# 說明
K-means 透過集群演算法將多維資料進行分群，但是K-means 不會告訴你該分幾群，所以可以通過手肘法（elbow method）跟輪廓係數法（Silhouette analysis）去協助選擇群數。

## K-means 步驟
1. 初始化 : 指定 K 個群，接著隨機挑選資料點當作該群中心
2. 分群(分配資料點) : 找出每個資料點距離最近的群中心
3. 重新分群 : 計算平均值，重新分配每一群的中心點
4. 結束 : 重複步驟2跟3，直到收斂

---

# 方法
## 1. 手肘法 Elbow method
手肘法是以誤差平方和（sum of the squared errors, SSE）作為指標，計算每一群中的每一個資料點到群中心的距離，找出 SSE 相對平緩的資料點作為拐點（Inflection point），並以此拐點選為群數。
SSE 計算方式如下：

![誤差平方和（sum of the squared errors, SSE](https://github.com/bill-git-code/kiwi/assets/63846041/ab6f19a8-207d-461d-82ef-cfe40b604021)

> 其中，

SSE 代表集群的好壞，也就是所有資料的誤差
- 總共有 K 個群
- Ci 代表其中一群，也就是第i個群
- p 代表 Ci 中的資料點
- mi 代表該群心，也就是 Ci中所有資料的平均值

---

## 2. 輪廓分析法 Silhouette analysis
輪廓係數法是判斷集群分析好壞的一種方法，目的是找出同一群的資料點內最近(凝聚度越小的值)，不同群越分散(分離度越高的值)，用來滿足集群主要的目標。
Silhouette 計算方式如下：

![Silhouette](https://github.com/bill-git-code/kiwi/assets/63846041/2ae3ef23-c9be-486a-ac6c-a203ae21e6a6)


> 其中，

S 代表集群的好壞，是所有資料的S(i)的平均值，S 值越大越好，越適合作為 K，範圍在[-1,1]之間
- S(i)越接近1，代表資料越適合此群，當凝聚度a(i)小於分離度b(i)，代表該群較為集中，並與不同群的距離較遠；
- S(i)越接近-1，代表資料越適合另一群；當凝聚度a(i)大於分離度b(i)，代表該群不集中，並與不同群的距離較近；
- S(i)越接近0，代表資料在兩個群的邊界上。
- 資料點的凝聚度a(i)， 代表與同一群的資料點的平均距離
- 資料點的分離度b(i) ，代表與不同群資料點的平均距離

---

# Code (Python)
- 這邊是 Python scikit-learn 實現這兩種方法的簡單範例:

## Step1. 安裝套件

```python!
import package
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
```


## Step2. 生成資料
- 使用sklearn中的聚類資料產生器make_blobs 去生成資料，根據你指定的特徵數量、中心數量、範圍來生成資料。

```python!
# 生成資料
X, y = make_blobs(n_samples=300, centers=4, random_state=42)
# 繪製散點圖
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('Generated Data with make_blobs(by kiwi_tech)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```
<div style="text-align: center;">
  <img src="https://github.com/bill-git-code/kiwi/assets/63846041/1f31b2e5-8361-4666-8c34-237da7ce21b2" alt="生成資料">
</div>


## Step3. 找出最佳的集群數
- 分別使用手肘法和輪廓分析法找到最佳的集群數，在這兩種情況下，可以觀察到對應了最佳的集群數的一個拐點:

### 1. 手肘法
- 通過繪製隨集群數增加的變化圖

```python!
# 使用手肘法找到最佳的集群數
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 繪製手肘法圖
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method(by kiwi_tech)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # WCSS代表群內平方和
plt.show()
```

- 從下圖可以發現 在n_clusters 為 4 時，出現拐點，集群的效果最好，所以可以選定4當作K。

<div style="text-align: center;">
  <img src="https://github.com/bill-git-code/kiwi/assets/63846041/b5c33ba0-bf4c-4961-b7a6-bb34eb0755e5" alt="手肘法">
</div>


### 2. 輪廓分析法
- 通過繪製輪廓分數隨集群數增加的變化圖

```python!
# 使用輪廓分析法找到最佳的集群數
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# 繪製輪廓分析法圖
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Analysis(by kiwi_tech)')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

```

- 從下圖可以發現 在n_clusters 為 4 時，集群的效果最好，所以可以選定4當作K
<div style="text-align: center;">
  <img src="https://github.com/bill-git-code/kiwi/assets/63846041/20b5f7db-fce8-49f4-9d2c-63f04414d020" alt="輪廓分析法">
</div>

---

sources
1. [sklearn.metrics: Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
2. [sklearn.metrics.silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

---
![KiwiTech_title_長條03](https://github.com/bill-git-code/kiwi/assets/63846041/51775980-7fa3-4490-9550-d35f23741244)


---
