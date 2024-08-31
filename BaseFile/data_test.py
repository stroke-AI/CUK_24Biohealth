import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, mannwhitneyu
from CNN_train import calculate_asymmetry_ratio, load_image_paths_and_labels

# 이미지 폴더 경로
normal_images_path = 'data/normal_pictures'
odd_images_path = 'data/odd_pictures'

# 이미지 경로와 레이블을 로드하는 함수 호출
image_paths, labels = load_image_paths_and_labels(odd_images_path, normal_images_path)

# 비대칭 지표를 계산하고 결과 리스트에 추가
results = []
for image_path in image_paths:
    asymmetry_ratio = calculate_asymmetry_ratio(image_path)
    if asymmetry_ratio is not None:
        group_label = "odd" if labels[image_paths.tolist().index(image_path)] == 1 else "normal"
        results.append({"Group": group_label, "Asymmetry": asymmetry_ratio})

# 데이터프레임 생성
df = pd.DataFrame(results)

# 두 그룹으로 분리
odd_group = df[df['Group'] == 'odd']['Asymmetry']
normal_group = df[df['Group'] == 'normal']['Asymmetry']

# 1. 정규성(Normality) 검정 - Shapiro-Wilk Test
shapiro_odd = shapiro(odd_group)
shapiro_normal = shapiro(normal_group)

print(f"Odd Group Shapiro-Wilk Test: W={shapiro_odd.statistic:.4f}, p-value={shapiro_odd.pvalue:.10f}")
print(f"Normal Group Shapiro-Wilk Test: W={shapiro_normal.statistic:.4f}, p-value={shapiro_normal.pvalue:.10f}")

# 2. 등분산성(Homogeneity of Variance) 검정 - Levene's Test
levene_test = levene(odd_group, normal_group)

print(f"Levene's Test for Equality of Variances: W={levene_test.statistic:.4f}, p-value={levene_test.pvalue:.10f}")

# 3. 비모수 검정 - Mann-Whitney U Test 수행
u_stat, p_value_mw = mannwhitneyu(odd_group, normal_group)

print(f"Mann-Whitney U Test: U-Statistic={u_stat:.4f}, p-value={p_value_mw:.10f}")

if p_value_mw < 0.05:
    print("두 그룹 간의 차이가 통계적으로 유의미합니다.")
else:
    print("두 그룹 간의 차이가 통계적으로 유의미하지 않습니다.")

# 4. 효과 크기 분석 (Cohen's d)
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return diff_mean / pooled_std

d = cohen_d(odd_group, normal_group)
print(f"Cohen's d: {d:.4f}")

# 5. 부트스트랩을 통한 신뢰 구간 계산
def bootstrap_confidence_interval(data, num_bootstrap=10000, ci=95):
    boot_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower_bound = np.percentile(boot_means, (100-ci)/2)
    upper_bound = np.percentile(boot_means, 100-(100-ci)/2)
    return lower_bound, upper_bound

ci_odd = bootstrap_confidence_interval(odd_group)
ci_normal = bootstrap_confidence_interval(normal_group)

print(f"Odd Group Mean Confidence Interval: {ci_odd}")
print(f"Normal Group Mean Confidence Interval: {ci_normal}")