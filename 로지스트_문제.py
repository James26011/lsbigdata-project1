# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# 1. 데이터 로드 및 로지스틱 회귀 모델 적합
# 데이터 생성

df = pd.read_csv('data/leukemia_remission.txt', delimiter='\t')


# 독립변수와 종속변수 분리
X = df[['CELL', 'SMEAR', 'INFIL', 'LI', 'BLAST', 'TEMP']]
y = df['REMISS']

# 상수항 추가
X = sm.add_constant(X)

# 로지스틱 회귀 모델 적합
model = sm.Logit(y, X)
result = model.fit()

# 회귀 결과 출력
print(result.summary())

# 2. 모델의 통계적 유의성 확인
# 유의성은 회귀모델의 p-value 및 로그가능도 검정 (likelihood ratio test)로 판단
p_value = result.llr_pvalue
print(f"모델의 유의성 p-value: {p_value}")


# 3. 통계적으로 유의한 변수 확인
# 각 변수의 p-value 확인
significant_vars = result.pvalues[result.pvalues < 0.2]
print(f"통계적으로 유의한 변수들: \n{significant_vars}")

# 4. 주어진 환자에 대한 오즈 계산
# 주어진 변수에 대한 값
new_patient = {
    'const': 1,
    'CELL': 0.65,
    'SMEAR': 0.45,
    'INFIL': 0.55,
    'LI': 1.2,
    'BLAST': 1.1,
    'TEMP': 0.9
}

# 오즈 계산
odds = np.exp(result.predict(pd.DataFrame([new_patient])))
print(f"해당 환자의 오즈: {odds[0]}")

# 5. 백혈병 세포 관측되지 않을 확률 계산
prob = odds / (1 + odds)
print(f"해당 환자의 백혈병 세포 관측되지 않을 확률: {prob[0]}")

# 6. TEMP 변수의 계수 및 치료 영향 설명
temp_coeff = result.params['TEMP']
print(f"TEMP 변수의 계수: {temp_coeff}")
print(f"TEMP 변수는 백혈병 세포 관측에 {temp_coeff}만큼 영향을 미침.")

# 7. CELL 변수의 99% 오즈비 신뢰구간 계산
conf = result.conf_int(alpha=0.01)
conf['OR'] = np.exp(result.params)
print(f"CELL 변수의 99% 오즈비 신뢰구간: {np.exp(conf.loc['CELL'])}")

# Beta__cell +- z0.005 *  52.135 
z005 = norm.ppf(0.995)  # 2.58
30.8301 - 2.58 * 52.135  # -103.678
30.8301 + 2.58 * 52.135  # 165.338


# 8. 예측 확률 및 혼동 행렬 구하기
pred_prob = result.predict(X)
y_pred = (pred_prob >= 0.5).astype(int)

# 혼동 행렬
cm = confusion_matrix(y, y_pred)
print(f"혼동 행렬:\n{cm}")

# 9. 모델의 Accuracy 계산
accuracy = accuracy_score(y, y_pred)
print(f"모델의 Accuracy: {accuracy}")

# 10. F1 Score 계산
f1 = f1_score(y, y_pred)
print(f"모델의 F1 Score: {f1}")
