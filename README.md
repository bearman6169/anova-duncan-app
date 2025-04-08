# ANOVA + Duncan 통계 분석 웹 앱

이 앱은 Streamlit을 기반으로 한 통계 분석 도구입니다.
사용자는 엑셀 파일을 업로드하여 그룹 간 차이를 분석할 수 있으며,
그룹이 2개일 경우 T-test, 3개 이상일 경우 One-way ANOVA 및 Duncan 사후검정을 자동으로 수행합니다.

## 주요 기능
- T-test 또는 One-way ANOVA 자동 수행
- Duncan's Multiple Range Test
- 통계 요약, 시각화 그래프, 결과 다운로드
- 엑셀 업로드 기반 자동 처리

## 사용법
1. 이 저장소를 Streamlit Cloud에 연결하거나 로컬에서 실행하세요.
2. 앱 실행:
```bash
streamlit run ANOVADUNCAN.py
```

## 필요 패키지
- streamlit
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels
- openpyxl