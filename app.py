import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import jieba
from collections import Counter

# ====================== 全局样式 ======================
st.set_page_config(page_title="舆情指挥中心·一体化大屏", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp {background: #0a1128; color: #e0e0e0;}
    .block-container {padding: 1rem 1.5rem; max-width: 100%;}
    h1, h2, h3, h4 {color: #e0f0ff !important; text-align: center; font-weight: bold;}
    div[data-testid="metric-container"] {background: #1a2446; border-radius: 8px; padding: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);}
    .stDivider {margin: 0.5rem 0;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%); border-right: 1px solid #2c3e50;}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {color: #ffffff !important; font-weight: 500 !important;}
    .stPlotlyChart {background: #0f1420; border-radius: 12px; padding: 8px;}
</style>
""", unsafe_allow_html=True)

# ====================== 词汇库 ======================
VOCAB_FILE = "vocab_lib.json"
DEFAULT_VOCAB = {
    "topic": {
        "亮证执法": ["亮证", "掏证", "出示证件", "警官证", "工作证", "执法证", "亮身份"],
        "特权争议": ["特权", "特权阶级", "特权思想", "滥用职权", "以权压人", "官威", "摆架子", "仗势欺人"],
        "基层执法": ["执法", "基层执法", "规范执法", "依法办事", "执法态度", "文明执法", "粗暴执法"],
        "行车纠纷": ["会车", "让路", "行车纠纷", "路怒", "开车斗气", "交通矛盾", "剐蹭"],
        "官方通报": ["官方", "通报", "调查结果", "回应", "处理结果", "公信力", "权威发布"],
        "舆情观望": ["吃瓜", "观望", "等反转", "理性看待", "不站队", "等真相", "客观评价"],
        "制度反思": ["权力约束", "监督机制", "作风整顿", "民心", "公平正义"]
    },
    "sentiment": {
        "强烈负面": ["愤怒", "心寒", "恶心", "离谱", "嚣张", "滥用职权", "傲慢", "双标", "官僚主义"],
        "一般负面": ["特权", "不爽", "不满", "反感", "质疑", "失望", "无语", "敷衍", "态度差"],
        "中性": ["吃瓜", "观望", "等通报", "理性", "客观", "不好评价", "事实说话", "中立"],
        "一般正面": ["理解", "认可", "支持", "合理", "规范", "辛苦", "不容易", "理性沟通"],
        "强烈正面": ["点赞", "公正", "给力", "值得肯定", "民心所向", "秉公执法", "规范到位"]
    }
}
def load_vocab():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return DEFAULT_VOCAB.copy()
def save_vocab(vocab):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
if "formal_vocab" not in st.session_state:
    st.session_state.formal_vocab = load_vocab()
if "test_vocab" not in st.session_state:
    st.session_state.test_vocab = {"topic": {}, "sentiment": {"强烈负面":[], "一般负面":[], "中性":[], "一般正面":[], "强烈正面":[]}}
if "temp_result" not in st.session_state:
    st.session_state.temp_result = None

def analyze(text):
    text = str(text).lower()
    all_topic = {**st.session_state.formal_vocab["topic"], **st.session_state.test_vocab["topic"]}
    topic = "其他"
    maxc = 0
    for t, wlist in all_topic.items():
        c = sum(1 for w in wlist if w in text)
        if c > maxc:
            maxc, topic = c, t
    all_sent = st.session_state.formal_vocab["sentiment"].copy()
    for k in all_sent.keys():
        all_sent[k] += st.session_state.test_vocab["sentiment"][k]
    score = 0.5
    for w in all_sent["强烈负面"]:
        if w in text: score -= 0.25
    for w in all_sent["一般负面"]:
        if w in text: score -= 0.12
    for w in all_sent["一般正面"]:
        if w in text: score += 0.08
    for w in all_sent["强烈正面"]:
        if w in text: score += 0.20
    score = round(max(0.1, min(0.9, score)), 2)
    if score <= 0.30:
        polar = "强烈负面"
    elif score <= 0.45:
        polar = "一般负面"
    elif score <= 0.55:
        polar = "中性"
    elif score <= 0.70:
        polar = "一般正面"
    else:
        polar = "强烈正面"
    return topic, score, polar

EMO_COLOR = {"强烈负面":"#D92121", "一般负面":"#FF7D45", "中性":"#FFD166", "一般正面":"#63D2FF", "强烈正面":"#28C76F"}

@st.cache_data
def load_comments_data():
    try:
        df = pd.read_csv("comments.csv", encoding="utf-8")
        if "text" not in df.columns:
            return pd.DataFrame()
        if "area" not in df.columns:
            df["area"] = "未标注"
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            df["time"] = pd.Timestamp.now()
        df["date"] = df["time"].dt.date
        df["hour"] = df["time"].dt.hour.fillna(12).astype(int)
        df["weekday"] = df["time"].dt.dayofweek
        analysis_results = df["text"].apply(lambda x: pd.Series(analyze(x)))
        df[["topic", "sentiment", "polar"]] = analysis_results
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_comments_data()
if df.empty:
    st.error("❌ 无法加载数据")
    st.stop()

# ====================== 侧边栏 ======================
with st.sidebar:
    st.markdown("# 📋 操作面板")
    st.markdown("---")
    st.markdown("### 📝 1. 新增评论")
    comment_text = st.text_area("✏️ 评论内容", height=100)
    col_date, col_area = st.columns(2)
    with col_date:
        comment_date = st.date_input("📅 日期", datetime.now())
    with col_area:
        area_list = sorted(df["area"].unique())
        comment_area = st.selectbox("📍 地区", area_list)
    col_hour, col_minute = st.columns(2)
    with col_hour:
        comment_hour = st.slider("🕐 小时", 0, 23, 12)
    with col_minute:
        comment_minute = st.slider("⏱️ 分钟", 0, 59, 30)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        test_btn = st.button("🔍 仅临时分析", use_container_width=True)
    with col_btn2:
        save_btn = st.button("💾 保存入库", use_container_width=True)
    if test_btn and comment_text:
        t, s, p = analyze(comment_text)
        st.session_state.temp_result = {"评论内容": comment_text[:50], "话题分类": t, "情感评分": s, "情感倾向": p}
        st.success(f"分析完成：话题【{t}】 | 倾向【{p}】 | 得分 {s}")
    if save_btn and comment_text:
        t, s, p = analyze(comment_text)
        comment_datetime = datetime(comment_date.year, comment_date.month, comment_date.day, comment_hour, comment_minute, 0)
        new_row = pd.DataFrame([{"time": comment_datetime, "text": comment_text, "area": comment_area}])
        existing_df = pd.read_csv("comments.csv", encoding="utf-8")
        df_new = pd.concat([existing_df, new_row], ignore_index=True)
        df_new.to_csv("comments.csv", index=False, encoding="utf-8-sig")
        st.success("✅ 评论已保存！")
        st.cache_data.clear()
        st.rerun()

st.title("🚨 网络舆情动态监控指挥中心 · 一体化大屏")
st.caption(f"数据时间：{df['date'].min()} 至 {df['date'].max()} | 共 {len(df)} 条评论")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("总评论量", f"{len(df):,}")
col2.metric("平均情感分", f"{df['sentiment'].mean():.3f}")
col3.metric("强烈负面", f"{(df['polar']=='强烈负面').mean()*100:.1f}%")
col4.metric("强烈正面", f"{(df['polar']=='强烈正面').mean()*100:.1f}%")
col5.metric("最热话题", df["topic"].value_counts().index[0])
col6.metric("评论高峰时段", f"{df['hour'].mode()[0]}:00")
st.divider()

# ====================== 舆情演变趋势 ======================
st.subheader("📈 舆情演变趋势分析")
daily_stats = df.groupby("date").agg(平均情感=("sentiment", "mean"),评论数量=("text", "count")).reset_index().sort_values("date")
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=daily_stats["date"], y=daily_stats["平均情感"], mode="lines+markers", name="日均情感分", line=dict(color="#00ff88", width=2)))
fig_trend.add_trace(go.Bar(x=daily_stats["date"], y=daily_stats["评论数量"], name="评论数量", yaxis="y2", marker_color="#8ab4f8", opacity=0.5))
fig_trend.update_layout(title="舆情情感演变趋势", xaxis_title="日期", yaxis_title="情感分", yaxis=dict(range=[0.1,0.9]), yaxis2=dict(title="评论数量", overlaying="y", side="right"), template="plotly_dark", height=450)
st.plotly_chart(fig_trend, use_container_width=True)
st.divider()

# ====================== 全国地图 ======================
st.subheader("🗺️ 全国各省舆情分布地图")
city_coords = {
    "北京":[116.40,39.90], "上海":[121.48,31.22], "广东":[113.23,23.16], "江苏":[118.78,32.04],
    "浙江":[120.15,30.28], "山东":[117.00,36.65], "四川":[104.06,30.67], "河南":[113.65,34.76],
    "湖北":[114.30,30.60], "湖南":[112.98,28.21], "陕西":[108.95,34.27], "重庆":[106.55,29.56]
}
province_sentiment = df.groupby("area")["sentiment"].mean().reset_index()
province_sentiment["lon"] = province_sentiment["area"].apply(lambda x: city_coords.get(x, [116.40,39.90])[0])
province_sentiment["lat"] = province_sentiment["area"].apply(lambda x: city_coords.get(x, [116.40,39.90])[1])
province_sentiment["评论量"] = df.groupby("area")["text"].count().reset_index()["text"]
col_map, col_bar = st.columns(2)
with col_map:
    fig_map = px.scatter_geo(province_sentiment, lon="lon", lat="lat", size="评论量", color="sentiment", hover_name="area", text="area", color_continuous_scale=["#D92121","#FF7D45","#FFD166","#63D2FF","#28C76F"], range_color=[0.2,0.8], size_max=40, title="中国舆情情感分布", template="plotly_dark", height=500)
    fig_map.update_geos(showcoastlines=True, coastlinecolor="gray", showland=True, landcolor="#1a1a2e", showocean=True, oceancolor="#0a1128")
    fig_map.update_layout(geo=dict(scope="asia", center=dict(lon=105, lat=35), lonaxis_range=[73,135], lataxis_range=[18,54]), height=500)
    st.plotly_chart(fig_map, use_container_width=True)
with col_bar:
    fig_bar = px.bar(province_sentiment.sort_values("sentiment"), x="area", y="sentiment", color="sentiment", color_continuous_scale=["#D92121","#FF7D45","#FFD166","#63D2FF","#28C76F"], title="各省平均情感得分", template="plotly_dark", height=500)
    fig_bar.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
st.divider()

# ====================== 时段分析 ======================
st.subheader("⏰ 时段评论热度分析")
hour_stats = df.groupby("hour")["text"].count().reset_index().rename(columns={"hour":"小时","text":"评论数"})
all_hours = pd.DataFrame({"小时":range(24)})
hour_stats = all_hours.merge(hour_stats, on="小时", how="left").fillna(0)
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    fig_hbar = px.bar(hour_stats, x="小时", y="评论数", title="各时段评论数量", color="评论数", color_continuous_scale="Viridis", template="plotly_dark", height=400)
    fig_hbar.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
    st.plotly_chart(fig_hbar, use_container_width=True)
with col_h2:
    hour_sent = df.groupby("hour")["sentiment"].mean().reset_index().rename(columns={"hour":"小时","sentiment":"平均情感分"})
    hour_sent = all_hours.merge(hour_sent, on="小时", how="left").fillna(0.5)
    fig_hline = px.line(hour_sent, x="小时", y="平均情感分", title="各时段平均情感", markers=True, template="plotly_dark", height=400)
    fig_hline.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_hline.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
    st.plotly_chart(fig_hline, use_container_width=True)
with col_h3:
    hour_polar = df.groupby(["hour","polar"]).size().reset_index(name="count")
    pivot = hour_polar.pivot(index="hour", columns="polar", values="count").fillna(0)
    for h in range(24):
        if h not in pivot.index:
            pivot.loc[h] = [0,0,0,0,0]
    pivot = pivot[["强烈负面","一般负面","中性","一般正面","强烈正面"]].sort_index()
    fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", title="时段×情感热力图", color_continuous_scale="Reds", template="plotly_dark", height=400)
    st.plotly_chart(fig_heat, use_container_width=True)
st.divider()

# ====================== 高频关键词表格（替代词云） ======================
st.subheader("📊 评论高频关键词")
stopwords = set(['的', '了', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们', '这', '那', '有', '在', '不', '也', '都', '说', '就', '要', '和', '与', '或', '但', '而', '并', '且', '如果', '虽然', '但是', '因为', '所以', '然后', '着', '过', '个', '种', '些', '能', '会', '可以', '没有', '还有', '或者', '而且', '所以', '这样', '那样', '怎么', '什么', '为什么', '哪里', '这里', '那里', '这个', '那个'])
# 对每条评论分词
word_list = []
for text in df["text"]:
    words = jieba.lcut(text)
    word_list.extend([w for w in words if len(w) >= 2 and w not in stopwords and not w.isdigit()])
word_freq = Counter(word_list).most_common(20)
freq_df = pd.DataFrame(word_freq, columns=["关键词", "出现次数"])
st.dataframe(freq_df, use_container_width=True)
st.caption("注：已过滤单字、数字和常见停用词，仅展示长度≥2的关键词。")

st.divider()

# ====================== 话题分析 ======================
st.subheader("💬 话题讨论分析")
topic_count = df["topic"].value_counts().reset_index()
topic_count.columns = ["话题", "评论数"]
fig_topic = px.bar(topic_count, x="话题", y="评论数", color="评论数", color_continuous_scale="Viridis", title="各话题讨论热度", template="plotly_dark", height=450)
fig_topic.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_topic, use_container_width=True)

# 话题×情感交叉表
cross = df.groupby(["topic", "polar"]).size().unstack(fill_value=0)
cross = cross[["强烈负面","一般负面","中性","一般正面","强烈正面"]]
fig_heat = px.imshow(cross, text_auto=True, aspect="auto", title="话题×情感交叉热力图", color_continuous_scale="RdBu_r", template="plotly_dark", height=450)
st.plotly_chart(fig_heat, use_container_width=True)
st.divider()

# ====================== 工作日 vs 周末 ======================
st.subheader("📅 工作日 vs 周末舆情对比")
df["is_weekend"] = df["weekday"].apply(lambda x: "周末" if x >= 5 else "工作日")
col_w1, col_w2 = st.columns(2)
with col_w1:
    week_comp = df.groupby("is_weekend").agg(评论数=("text","count"),情感分=("sentiment","mean")).reset_index()
    fig_week = px.bar(week_comp, x="is_weekend", y="评论数", title="评论数量对比", color="is_weekend", color_discrete_map={"工作日":"#4285f4","周末":"#ea4335"}, template="plotly_dark", height=400)
    st.plotly_chart(fig_week, use_container_width=True)
with col_w2:
    week_polar = df.groupby(["is_weekend","polar"]).size().reset_index(name="数量")
    fig_polar = px.bar(week_polar, x="is_weekend", y="数量", color="polar", barmode="group", color_discrete_map=EMO_COLOR, title="情感分布对比", template="plotly_dark", height=400)
    st.plotly_chart(fig_polar, use_container_width=True)
st.divider()

# ====================== 核心洞察 ======================
st.subheader("🎯 舆情核心洞察与建议")
col_i1, col_i2, col_i3 = st.columns(3)
with col_i1:
    peak_hour = hour_stats.loc[hour_stats["评论数"].idxmax(), "小时"]
    st.metric("评论高峰时段", f"{int(peak_hour)}:00")
with col_i2:
    morning = hour_sent[hour_sent["小时"].between(6,11)]["平均情感分"].mean()
    night = hour_sent[hour_sent["小时"].between(18,23)]["平均情感分"].mean()
    st.metric("夜间情感", f"{night:.3f}", delta=f"{night-morning:.3f}")
with col_i3:
    neg_hour = df[df["polar"].isin(["强烈负面","一般负面"])].groupby("hour").size()
    if not neg_hour.empty:
        st.warning(f"负面高峰时段：{neg_hour.idxmax()}:00")
st.divider()

# ====================== 临时结果 & 数据预览 ======================
if st.session_state.temp_result:
    st.subheader("📋 最新评论分析结果")
    st.json(st.session_state.temp_result)

with st.expander("📊 查看原始数据预览"):
    st.dataframe(df[["time","text","area","topic","polar","sentiment"]].head(100), use_container_width=True)
