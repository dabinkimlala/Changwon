# ===== 1. 라이브러리 불러오기 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from gurobipy import Model, GRB, quicksum
import folium
import warnings
warnings.filterwarnings("ignore")

# 운영체제별 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac: AppleGothic 사용
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows: Malgun Gothic 사용
plt.rc('axes', unicode_minus=False)  # 마이너스 폰트 깨짐 방지 설정
%matplotlib inline
%config InlineBackend.figure_format = 'retina'  # 그래프를 더 선명하게 출력

# ===== 2. 데이터 불러오기 및 간단한 EDA =====
# 데이터 로드
df = pd.read_csv('/content/창원시 데이터 프레임 최종 찐찐찐찐찐.csv', encoding='utf-8')

# 데이터 전처리: 결측값 처리
df = df.fillna(0)

# 데이터 확인
print("===== 데이터 요약 =====")
print(df.head())  # 데이터 상위 5개 확인
print("\n결측값 확인:\n", df.isnull().sum())  # 결측값 확인

# ===== 3. 데이터 표준화 =====
# 군집화에 사용할 주요 변수 선택
features = ['Inflow and outflow', 'floating population', 'Number of storage units',
            'the resident population', 'Number of terminals', 'School', 'park',
            'Amount used', 'buscount']

# 데이터 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[features])

# ===== 4. PCA (주성분 분석) =====
# PCA 수행
pca = PCA()
data_pca = pca.fit_transform(data_scaled)

# 누적 분산 비율 시각화
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow Method for Determining Number of Components')
plt.grid()
plt.show()

# ===== 5. K-Means 군집화 =====
# 최적 클러스터 수 확인 (실루엣 점수 기반)
def visualize_silhouette_layer(data):
    clusters_range = range(2, 10)
    results = []
    for i in clusters_range:
        km = KMeans(n_clusters=i, random_state=42)
        cluster_labels = km.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])
    
    # 결과 시각화
    result_df = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    sns.heatmap(result_df.set_index('n_clusters'), annot=True, fmt='.3f', cmap='RdYlGn')
    plt.title("Silhouette Scores by Cluster Number")
    plt.show()

visualize_silhouette_layer(data_scaled)

# 최적 클러스터 수 = 4로 가정하여 군집화 수행
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# 결과 시각화 (PCA 2차원으로 변환 후)
data_scaled_df = pd.DataFrame(data_scaled, columns=['PC' + str(i + 1) for i in range(data_scaled.shape[1])])
data_scaled_df["Cluster"] = clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data_scaled_df, x='PC1', y='PC2', hue='Cluster', palette='Set3')
plt.title('K-Means Clustering (2D)')
plt.show()

# ===== 6. GMM (Gaussian Mixture Model) =====
# GMM 군집화 수행
gmm = GaussianMixture(n_components=4, random_state=0)
gmm_clusters = gmm.fit_predict(data_scaled)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=gmm_clusters, cmap='viridis')
plt.title('GMM Clustering (2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()

# ===== 7. 계층적 군집화 (Hierarchical Clustering) =====
# 덴드로그램 생성
linked = linkage(data_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# 계층적 군집화 수행
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
hier_clusters = agg_clustering.fit_predict(data_scaled)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=hier_clusters, cmap='rainbow')
plt.title('Hierarchical Clustering (2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# ===== 8. MCLP (Maximal Covering Location Problem) =====
# Haversine 함수 정의 (거리 계산용)
def haversine(point1, point2):
    lon1, lat1 = point1
    lon2, lat2 = point2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a)) * 1000  # Return distance in meters

# MCLP 모델 정의
def mclp(sites, demand_points, K, radius):
    m = Model()
    y = {j: m.addVar(vtype=GRB.BINARY, name=f"y{j}") for j in range(len(sites))}
    for i, point in enumerate(demand_points):
        m.addConstr(quicksum(y[j] for j in np.where([haversine(point, site) <= radius for site in sites])[0]) >= 1)
    m.setObjective(quicksum(y[j] for j in range(len(sites))), GRB.MAXIMIZE)
    m.optimize()
    return [sites[j] for j in range(len(sites)) if y[j].X > 0]

# 예제 데이터: sites와 demand_points는 실제 데이터로 대체
sites = [[128.7, 35.5], [128.8, 35.6], [128.6, 35.4]]
demand_points = [[128.71, 35.51], [128.72, 35.52], [128.73, 35.53]]
optimal_sites = mclp(sites, demand_points, K=2, radius=500)

# 결과 시각화 (Folium 사용)
m = folium.Map(location=optimal_sites[0], zoom_start=13)
for location in demand_points:
    folium.Marker(location, icon=folium.Icon(color="green")).add_to(m)
for location in optimal_sites:
    folium.Circle(location=location, radius=500, color="blue", fill=True).add_to(m)

# 결과 저장
m.save('mclp_result.html')
