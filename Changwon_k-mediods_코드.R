# 라이브러리 로드
library(cluster)
library(tidyverse)

# 데이터 불러오기
df <- read.csv("C:/Users/PC/Downloads/data_pca.csv")

# K-medoids 클러스터링 수행 (k=4, 표준화 적용)
set.seed(123)
pam.result <- pam(df, k = 4, stand = TRUE)

# 클러스터링 시각화
clusplot(pam.result, main = 'K-medoids Clustering', labels = 2, cex = 3)

# 클러스터 결과 저장
df$cluster <- pam.result$clustering

# 실루엣 정보 확인
pam.result$silinfo

# 결과 데이터 저장
write_csv(df, "k-medoids_df.csv")

# 클러스터 추가된 데이터 미리보기
head(df)
