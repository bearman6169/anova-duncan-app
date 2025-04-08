# 📦 필요한 라이브러리 설치 안내
# 아래 명령어를 실행하여 필요한 패키지를 설치하세요:
# pip install streamlit pandas matplotlib seaborn scipy statsmodels openpyxl

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# Duncan test 함수 정의 (직접 구현)
def duncan_test(df, group_col, value_col, alpha=0.05):
    groups = sorted(df[group_col].unique())
    k = len(groups)
    group_stats = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    group_stats = group_stats.sort_values('mean', ascending=False)
    formula = f'{value_col} ~ C({group_col})'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    MSE = anova_table['sum_sq'][1] / anova_table['df'][1]
    df_error = df.shape[0] - k
    p_values = np.arange(2, k+1)
    q_values = [stats.studentized_range.ppf(1-alpha, p, df_error) for p in p_values]
    LSR = [q * np.sqrt(MSE / stats.hmean(group_stats['count'])) for q in q_values]
    means = group_stats['mean'].values
    groups_sorted = group_stats[group_col].values
    significance_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i+1, k):
            mean_diff = abs(means[i] - means[j])
            p = j - i + 1
            if mean_diff > LSR[p-2]:
                significance_matrix[i, j] = 1
                significance_matrix[j, i] = 1
    labels = ['' for _ in range(k)]
    current_label = 97
    for i in range(k):
        if labels[i] != '':
            continue
        labels[i] = chr(current_label)
        for j in range(i+1, k):
            if significance_matrix[i, j] == 0:
                labels[j] += chr(current_label)
        current_label += 1
    result_df = pd.DataFrame({
        'group': groups_sorted,
        'mean': means,
        'n': group_stats['count'].values,
        'std': group_stats['std'].values,
        'group_label': labels
    })
    return result_df

st.set_page_config(page_title="통계 분석 도구", layout="centered")
st.title("🥩 Meat Quality Statistical Analyzer")

menu = st.sidebar.radio("메뉴를 선택하세요:", ["📊 통계 분석 시작하기", "❌ 종료하기"])

if menu == "❌ 종료하기":
    st.warning("프로그램이 종료되었습니다.")
    st.stop()

if menu == "📊 통계 분석 시작하기":
    st.subheader("1️⃣ 엑셀 파일 업로드")
    uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            if df.shape[1] < 2:
                st.error("엑셀 파일은 최소 두 개의 열(그룹, 값)로 구성되어야 합니다.")
                st.stop()

            df.columns = ["Group", "Value"]
            df.dropna(inplace=True)
            group_count = df["Group"].nunique()
            obs_count = len(df)

            tab1, tab2, tab3 = st.tabs(["📋 분석 요약", "📊 그래프 보기", "📁 결과 저장"])

            with tab1:
                st.subheader("📋 분석 요약")
                st.write(f"총 그룹 수: {group_count}")
                st.write(f"총 관측값 수: {obs_count}")

                if group_count < 2:
                    st.warning("두 개 이상의 그룹이 필요합니다.")
                    st.stop()

                elif group_count == 2:
                    group_vals = df.groupby("Group")["Value"]
                    values = [group_vals.get_group(g).values for g in group_vals.groups]
                    stat, pval = stats.ttest_ind(*values, equal_var=False)
                    st.markdown(f"**T-test 결과**: p-value = `{pval:.4f}`")
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    st.markdown(f"**유의성 표시**: `{stars}`")
                    duncan_result = None
                else:
                    model = ols('Value ~ C(Group)', data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    pval = anova_table["PR(>F)"].values[0]
                    st.markdown("**One-way ANOVA 결과**")
                    st.dataframe(anova_table.round(4))
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    st.markdown(f"**p-value**: `{pval:.4f}` → `{stars}`")

                    duncan_result = None
                    if pval < 0.05:
                        duncan_result = duncan_test(df, 'Group', 'Value')
                        st.markdown("**Duncan 사후검정 결과**")
                        st.dataframe(duncan_result.round(3))

            with tab2:
                st.subheader("📊 그룹별 시각화")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="Group", y="Value", data=df, ax=ax, palette="Set3")
                sns.stripplot(x="Group", y="Value", data=df, ax=ax, color="black", alpha=0.5, jitter=True)

                group_means = df.groupby("Group")["Value"].mean()
                group_sds = df.groupby("Group")["Value"].std()
                positions = range(len(group_means))

                for i, group in zip(positions, group_means.index):
                    mean = group_means[group]
                    sd = group_sds[group]
                    ax.text(i, mean + sd + 0.2, f"{mean:.2f}±{sd:.2f}", ha='center', fontsize=9)

                if pval < 0.05:
                    ax.text(len(group_means) / 2 - 0.5, df["Value"].max() + 0.3, stars, ha='center', fontsize=16, color="red")

                st.pyplot(fig)

            with tab3:
                st.subheader("📁 결과 저장")
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ 원본 데이터 다운로드 (.csv)", csv, "data.csv", "text/csv")

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f"<a href='data:image/png;base64,{b64}' download='plot.png'>🖼️ 그래프 이미지 다운로드</a>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"파일을 처리하는 중 오류 발생: {e}")
