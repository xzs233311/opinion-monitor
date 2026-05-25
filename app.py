import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# ====================== 全局大屏样式 ======================
st.set_page_config(page_title="舆情指挥中心·一体化大屏", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp {background: #0a1128; color: #e0e0e0;}
    .block-container {padding: 1rem 1.5rem; max-width: 100%;}
    h1, h2, h3, h4 {color: #e0f0ff !important; text-align: center; font-weight: bold;}
    div[data-testid="metric-container"] {background: #1a2446; border-radius: 8px; padding: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);}
    .stDivider {margin: 0.5rem 0;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
        border-right: 1px solid #2c3e50;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4ecdc4 !important;
        border-bottom: 2px solid #4ecdc4;
        padding-bottom: 5px;
    }
    [data-testid="stSidebar"] .stButton button {
        background: #2c3e50;
        color: white;
        border-radius: 20px;
        transition: all 0.3s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #4ecdc4;
        color: #0a1128;
    }
    
    .stPlotlyChart {
        background: #0f1420;
        border-radius: 12px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ====================== 词汇库 ======================
VOCAB_FILE = "vocab_lib.json"

DEFAULT_VOCAB = {
    "topic": {
        "亮证执法": ["亮证", "掏证", "出示证件", "警官证", "工作证", "执法证", "亮身份", "公职身份", "证件压人"],
        "特权争议": ["特权", "特权阶级", "特权思想", "搞特权", "滥用职权", "以权压人", "官威", "摆架子", "官本位", "仗势欺人", "高人一等"],
        "基层执法": ["执法", "基层执法", "规范执法", "依法办事", "执法态度", "文明执法", "粗暴执法", "一线人员"],
        "行车纠纷": ["会车", "让路", "行车纠纷", "路怒", "开车斗气", "交通矛盾", "剐蹭", "别车", "抢道"],
        "官方通报": ["官方", "通报", "调查结果", "回应", "处理结果", "公信力", "权威发布", "官方声明"],
        "舆情观望": ["吃瓜", "观望", "等反转", "理性看待", "不站队", "等真相", "客观评价", "中立"],
        "制度反思": ["权力约束", "监督机制", "作风整顿", "民心", "公平正义", "制度建设"]
    },
    "sentiment": {
        "强烈负面": ["啧啧啧", "呵呵", "阴阳怪气", "讽刺", "嘲讽", "愤怒", "心寒", "恶心", "离谱", "嚣张", "滥用职权", "傲慢", "双标", "践踏公平", "仗势欺人", "官僚主义", "气死", "痛恨", "绝望", "寒心"],
        "一般负面": ["特权", "不爽", "不满", "反感", "质疑", "失望", "无语", "敷衍", "态度差", "摆架子", "偏心", "不公正", "吐槽", "无奈", "憋屈", "委屈"],
        "中性": ["吃瓜", "观望", "等通报", "理性", "客观", "不好评价", "事实说话", "等待后续", "不站队", "中立看待", "围观", "蹲后续"],
        "一般正面": ["理解", "认可", "支持", "合理", "规范", "辛苦", "不容易", "理性沟通", "正常处理", "可以接受", "还好", "不错"],
        "强烈正面": ["点赞", "公正", "给力", "值得肯定", "民心所向", "秉公执法", "规范到位", "为民服务", "靠谱", "三观正", "大快人心"]
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
    st.session_state.test_vocab = {
        "topic": {},
        "sentiment": {"强烈负面":[], "一般负面":[], "中性":[], "一般正面":[], "强烈正面":[]}
    }
if "temp_result" not in st.session_state:
    st.session_state.temp_result = None

# ====================== 情感分析函数 ======================
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

EMO_COLOR = {
    "强烈负面":"#D92121",
    "一般负面":"#FF7D45",
    "中性":"#FFD166",
    "一般正面":"#63D2FF",
    "强烈正面":"#28C76F"
}

# ====================== 读取评论数据 ======================
@st.cache_data
def load_comments_data():
    try:
        df = pd.read_csv("comments.csv", encoding="utf-8")
        
        if "text" not in df.columns:
            st.error("comments.csv必须包含'text'列")
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
        df["weekday_name"] = df["time"].dt.day_name()
        
        analysis_results = df["text"].apply(lambda x: pd.Series(analyze(x)))
        df[["topic", "sentiment", "polar"]] = analysis_results
        
        return df
    except Exception as e:
        st.error(f"加载comments.csv失败: {e}")
        return pd.DataFrame()

df = load_comments_data()

if df.empty:
    st.error("❌ 无法加载数据，请检查comments.csv文件格式")
    st.stop()

# ====================== 中国地图数据（使用经纬度标点方式，100%可靠） ======================
# 中国主要城市的经纬度
city_coords = {
    "北京": [116.40, 39.90], "上海": [121.48, 31.22], "广东": [113.23, 23.16], "江苏": [118.78, 32.04],
    "浙江": [120.15, 30.28], "山东": [117.00, 36.65], "四川": [104.06, 30.67], "河南": [113.65, 34.76],
    "湖北": [114.30, 30.60], "湖南": [112.98, 28.21], "陕西": [108.95, 34.27], "重庆": [106.55, 29.56],
    "天津": [117.20, 39.13], "福建": [119.30, 26.08], "安徽": [117.27, 31.86], "河北": [114.48, 38.03],
    "辽宁": [123.43, 41.80], "江西": [115.89, 28.68], "云南": [102.71, 25.05], "广西": [108.32, 22.84],
    "山西": [112.56, 37.87], "贵州": [106.71, 26.57], "黑龙江": [126.64, 45.76], "吉林": [125.35, 43.88],
    "甘肃": [103.82, 36.06], "海南": [110.35, 20.02], "宁夏": [106.27, 38.47], "青海": [101.78, 36.62],
    "西藏": [91.14, 29.66], "新疆": [87.62, 43.83], "内蒙古": [111.65, 40.82], "香港": [114.17, 22.27],
    "澳门": [113.54, 22.19], "台湾": [121.52, 25.03]
}

# 计算各省情感得分
province_sentiment = df.groupby("area")["sentiment"].mean().reset_index()
province_sentiment["lon"] = province_sentiment["area"].apply(lambda x: city_coords.get(x, [116.40, 39.90])[0])
province_sentiment["lat"] = province_sentiment["area"].apply(lambda x: city_coords.get(x, [116.40, 39.90])[1])
province_sentiment["评论量"] = df.groupby("area")["text"].count().reset_index()["text"]
province_sentiment["情感描述"] = province_sentiment["sentiment"].apply(
    lambda x: "强烈负面" if x < 0.3 else ("一般负面" if x < 0.45 else ("中性" if x < 0.55 else ("一般正面" if x < 0.7 else "强烈正面")))
)

# ====================== 侧边栏 ======================
with st.sidebar:
    st.markdown("# 📋 操作面板")
    st.markdown("---")
    
    st.markdown("### 📝 1. 新增评论")
    comment_text = st.text_area("✏️ 评论内容", height=100, placeholder="例如：这个事件太离谱了，特权思想必须整治！")
    
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
        st.session_state.temp_result = {"评论内容": comment_text[:50] + "...", "话题分类": t, "情感评分": s, "情感倾向": p}
        st.success(f"📊 分析完成：话题【{t}】 | 倾向【{p}】 | 得分 {s}")
    
    if save_btn and comment_text:
        t, s, p = analyze(comment_text)
        comment_datetime = datetime(comment_date.year, comment_date.month, comment_date.day, comment_hour, comment_minute, 0)
        new_row = pd.DataFrame([{"time": comment_datetime, "text": comment_text, "area": comment_area}])
        existing_df = pd.read_csv("comments.csv", encoding="utf-8")
        df_new = pd.concat([existing_df, new_row], ignore_index=True)
        df_new.to_csv("comments.csv", index=False, encoding="utf-8-sig")
        st.success("✅ 评论已保存！页面即将刷新...")
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🔧 2. 扩充关键词库")
    vocab_type = st.radio("📌 选择词汇类型", ["🏷️ 话题关键词", "😊 情感关键词"], horizontal=True)
    
    if "话题" in vocab_type:
        topic_name = st.text_input("话题名称", placeholder="例如：网络暴力")
        topic_word = st.text_input("关键词列表", placeholder="例如：网暴,键盘侠,人身攻击")
        col_test, col_save = st.columns(2)
        with col_test:
            test_topic_btn = st.button("🧪 测试添加", use_container_width=True)
        with col_save:
            save_topic_btn = st.button("💾 正式添加", use_container_width=True)
        if test_topic_btn and topic_name and topic_word:
            words = [w.strip() for w in topic_word.split(",")]
            st.session_state.test_vocab["topic"][topic_name] = words
            st.success(f"✅ 测试添加成功：{topic_name}")
        if save_topic_btn and topic_name and topic_word:
            words = [w.strip() for w in topic_word.split(",")]
            st.session_state.formal_vocab["topic"][topic_name] = words
            save_vocab(st.session_state.formal_vocab)
            st.success(f"✅ 正式添加成功：{topic_name}")
    else:
        sent_level = st.selectbox("情感层级", ["强烈负面", "一般负面", "中性", "一般正面", "强烈正面"])
        sent_word = st.text_input("关键词列表", placeholder="例如：离谱,嚣张,傲慢")
        col_test, col_save = st.columns(2)
        with col_test:
            test_sent_btn = st.button("🧪 测试添加", use_container_width=True)
        with col_save:
            save_sent_btn = st.button("💾 正式添加", use_container_width=True)
        if test_sent_btn and sent_word:
            words = [w.strip() for w in sent_word.split(",")]
            st.session_state.test_vocab["sentiment"][sent_level].extend(words)
            st.success(f"✅ 测试添加成功：{sent_level}")
        if save_sent_btn and sent_word:
            words = [w.strip() for w in sent_word.split(",")]
            st.session_state.formal_vocab["sentiment"][sent_level].extend(words)
            save_vocab(st.session_state.formal_vocab)
            st.success(f"✅ 正式添加成功：{sent_level}")
    
    st.markdown("---")
    
    st.markdown("### 🗑️ 3. 词汇库管理")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🧹 清空测试词库", use_container_width=True):
            st.session_state.test_vocab = {"topic": {}, "sentiment":{"强烈负面":[], "一般负面":[], "中性":[], "一般正面":[], "强烈正面":[]}}
            st.success("✅ 测试词库已清空")
    with col_reset2:
        if st.button("🔄 重置正式词库", use_container_width=True):
            st.session_state.formal_vocab = DEFAULT_VOCAB.copy()
            save_vocab(st.session_state.formal_vocab)
            st.success("✅ 正式词库已恢复默认")
    
    st.markdown("---")
    
    st.markdown("### 📊 4. 数据概览")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric("总评论数", len(df))
    with col_stats2:
        st.metric("话题数量", df["topic"].nunique())

# ====================== 主大屏 ======================
st.title("🚨 网络舆情动态监控指挥中心 · 一体化大屏")
st.caption(f"📅 数据时间范围：{df['date'].min()} 至 {df['date'].max()} | 共 {len(df)} 条评论")

# ====================== 第一行：核心KPI ======================
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("📊 总评论量", f"{len(df):,}")
col2.metric("📈 平均情感分", f"{df['sentiment'].mean():.3f}")
col3.metric("🔴 强烈负面", f"{(df['polar']=='强烈负面').mean()*100:.1f}%")
col4.metric("🟢 强烈正面", f"{(df['polar']=='强烈正面').mean()*100:.1f}%")
col5.metric("🔥 最热话题", df["topic"].value_counts().index[0])
col6.metric("⏰ 评论高峰时段", f"{df['hour'].mode()[0]}:00")
st.divider()

# ====================== 第二行：舆情演变趋势分析 ======================
st.subheader("📈 舆情演变趋势分析")

daily_stats = df.groupby("date").agg(
    平均情感=("sentiment", "mean"),
    评论数量=("text", "count")
).reset_index().sort_values("date")

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=daily_stats["date"], y=daily_stats["平均情感"], 
                               mode="lines+markers", name="日均情感分",
                               line=dict(color="#00ff88", width=2), marker=dict(size=6)))
fig_trend.add_trace(go.Bar(x=daily_stats["date"], y=daily_stats["评论数量"], 
                           name="评论数量", yaxis="y2", marker_color="#8ab4f8", opacity=0.5))
fig_trend.update_layout(title="舆情情感演变趋势", xaxis_title="日期", 
                       yaxis_title="情感分（越高越正面）", yaxis=dict(range=[0.1, 0.9]),
                       yaxis2=dict(title="评论数量", overlaying="y", side="right"),
                       template="plotly_dark", height=500, hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)
st.divider()

# ====================== 第三行：中国地图（散点图方式，100%可靠） + 地区柱状图 ======================
st.subheader("🗺️ 全国各省舆情分布地图")

col_map, col_bar = st.columns(2)

with col_map:
    # 使用 scatter_geo 制作中国地图散点图（这种方式100%可靠）
    fig_scatter_map = px.scatter_geo(
        province_sentiment,
        lon="lon",
        lat="lat",
        size="评论量",
        color="sentiment",
        hover_name="area",
        text="area",
        color_continuous_scale=["#D92121", "#FF7D45", "#FFD166", "#63D2FF", "#28C76F"],
        range_color=[0.2, 0.8],
        size_max=40,
        title="🇨🇳 中国舆情情感分布地图",
        labels={"sentiment": "情感得分", "area": "省份", "评论量": "评论数量"},
        template="plotly_dark",
        height=550
    )
    fig_scatter_map.update_geos(
        showcoastlines=True, coastlinecolor="gray",
        showland=True, landcolor="#1a1a2e",
        showocean=True, oceancolor="#0a1128",
        showcountries=True, countrycolor="#2c3e50",
        projection_type="natural earth"
    )
    fig_scatter_map.update_layout(
        geo=dict(
            scope="asia",
            center=dict(lon=105, lat=35),
            lonaxis_range=[73, 135],
            lataxis_range=[18, 54]
        ),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_scatter_map, use_container_width=True)
    
    # 地图说明
    st.caption("💡 地图说明：🔴红点=负面 🟡黄点=中性 🟢绿点=正面 | 点越大=评论量越多")

with col_bar:
    # 地区情感柱状图
    region_sentiment = df.groupby("area")["sentiment"].mean().sort_values().reset_index()
    fig_region = px.bar(region_sentiment, x="area", y="sentiment", 
                        color="sentiment", color_continuous_scale=["#D92121", "#FF7D45", "#FFD166", "#63D2FF", "#28C76F"],
                        title="各省平均情感得分",
                        labels={"sentiment": "情感得分", "area": "省份"},
                        template="plotly_dark", height=550)
    fig_region.update_layout(xaxis_tickangle=-45)
    fig_region.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="中性线")
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ====================== 第四行：地区情感构成 ======================
region_polar = df.groupby(["area", "polar"]).size().reset_index(name="数量")
top_areas = region_polar.groupby("area")["数量"].sum().sort_values(ascending=False).head(8).index
region_polar_top = region_polar[region_polar["area"].isin(top_areas)]
fig_stacked = px.bar(region_polar_top, x="area", y="数量", color="polar",
                     color_discrete_map=EMO_COLOR, title="主要地区情感构成",
                     template="plotly_dark", height=450)
fig_stacked.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_stacked, use_container_width=True)
st.divider()

# ====================== 第五行：时段分析 ======================
st.subheader("⏰ 时段评论热度分析")

col_hour1, col_hour2, col_hour3 = st.columns(3)

with col_hour1:
    hour_stats = df.groupby("hour")["text"].count().reset_index()
    hour_stats.columns = ["小时", "评论数"]
    all_hours = pd.DataFrame({"小时": range(24)})
    hour_stats = all_hours.merge(hour_stats, on="小时", how="left").fillna(0)
    
    fig_hour_bar = px.bar(hour_stats, x="小时", y="评论数", 
                          title="各时段评论数量分布",
                          labels={"小时": "时间（点）", "评论数": "评论数量"},
                          color="评论数", color_continuous_scale="Viridis",
                          template="plotly_dark", height=400)
    fig_hour_bar.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
    st.plotly_chart(fig_hour_bar, use_container_width=True)

with col_hour2:
    hour_sentiment = df.groupby("hour")["sentiment"].mean().reset_index()
    hour_sentiment.columns = ["小时", "平均情感分"]
    hour_sentiment = all_hours.merge(hour_sentiment, on="小时", how="left").fillna(0.5)
    
    fig_hour_line = px.line(hour_sentiment, x="小时", y="平均情感分",
                            title="各时段平均情感得分",
                            labels={"小时": "时间（点）", "平均情感分": "情感得分"},
                            markers=True, template="plotly_dark", height=400)
    fig_hour_line.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="中性线")
    fig_hour_line.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
    st.plotly_chart(fig_hour_line, use_container_width=True)

with col_hour3:
    hour_polar = df.groupby(["hour", "polar"]).size().reset_index(name="数量")
    hour_pivot = hour_polar.pivot(index="hour", columns="polar", values="数量").fillna(0)
    expected_columns = ["强烈负面", "一般负面", "中性", "一般正面", "强烈正面"]
    for col in expected_columns:
        if col not in hour_pivot.columns:
            hour_pivot[col] = 0
    hour_pivot = hour_pivot[expected_columns]
    for h in range(24):
        if h not in hour_pivot.index:
            hour_pivot.loc[h] = [0, 0, 0, 0, 0]
    hour_pivot = hour_pivot.sort_index()
    
    fig_heatmap = px.imshow(hour_pivot, text_auto=True, aspect="auto",
                            title="时段×情感分布热力图",
                            labels={"x": "情感倾向", "y": "小时"},
                            color_continuous_scale="Reds",
                            template="plotly_dark", height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
st.divider()

# ====================== 第六行：工作日vs周末 ======================
st.subheader("📅 工作日 vs 周末舆情对比")

df["is_weekend"] = df["weekday"].apply(lambda x: "周末" if x >= 5 else "工作日")

col_week1, col_week2 = st.columns(2)

with col_week1:
    week_comparison = df.groupby("is_weekend").agg(
        评论数量=("text", "count"),
        平均情感=("sentiment", "mean")
    ).reset_index()
    
    fig_week_bar = px.bar(week_comparison, x="is_weekend", y="评论数量", 
                          title="评论数量对比", color="is_weekend",
                          color_discrete_map={"工作日": "#4285f4", "周末": "#ea4335"},
                          template="plotly_dark", height=400)
    st.plotly_chart(fig_week_bar, use_container_width=True)

with col_week2:
    week_polar = df.groupby(["is_weekend", "polar"]).size().reset_index(name="数量")
    fig_week_polar = px.bar(week_polar, x="is_weekend", y="数量", color="polar",
                            barmode="group", color_discrete_map=EMO_COLOR,
                            title="情感分布对比", template="plotly_dark", height=400)
    st.plotly_chart(fig_week_polar, use_container_width=True)

# 工作日vs周末时段分布
weekday_hour = df[df["is_weekend"] == "工作日"].groupby("hour")["text"].count().reset_index()
weekday_hour.columns = ["小时", "工作日评论数"]
weekend_hour = df[df["is_weekend"] == "周末"].groupby("hour")["text"].count().reset_index()
weekend_hour.columns = ["小时", "周末评论数"]

hour_compare = all_hours.merge(weekday_hour, on="小时", how="left").fillna(0)
hour_compare = hour_compare.merge(weekend_hour, on="小时", how="left").fillna(0)

fig_week_compare = go.Figure()
fig_week_compare.add_trace(go.Bar(x=hour_compare["小时"], y=hour_compare["工作日评论数"], name="工作日", marker_color="#4285f4"))
fig_week_compare.add_trace(go.Bar(x=hour_compare["小时"], y=hour_compare["周末评论数"], name="周末", marker_color="#ea4335"))
fig_week_compare.update_layout(title="工作日 vs 周末 时段评论分布对比", 
                               xaxis_title="小时", yaxis_title="评论数量",
                               barmode="group", template="plotly_dark", height=450)
st.plotly_chart(fig_week_compare, use_container_width=True)
st.divider()

# ====================== 第七行：话题分析 ======================
st.subheader("💬 话题讨论分析")
col_topic1, col_topic2 = st.columns(2)

with col_topic1:
    topic_count = df["topic"].value_counts().reset_index()
    topic_count.columns = ["话题", "评论数"]
    fig_topic = px.bar(topic_count.head(10), x="话题", y="评论数", 
                       color="评论数", color_continuous_scale="Viridis",
                       title="各话题讨论热度", template="plotly_dark", height=450)
    fig_topic.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_topic, use_container_width=True)

with col_topic2:
    cross_data = df.groupby(["topic", "polar"]).size().reset_index(name="count")
    cross_pivot = cross_data.pivot(index="topic", columns="polar", values="count").fillna(0)
    expected_cols = ["强烈负面", "一般负面", "中性", "一般正面", "强烈正面"]
    for col in expected_cols:
        if col not in cross_pivot.columns:
            cross_pivot[col] = 0
    cross_pivot = cross_pivot[expected_cols]
    fig_heat = px.imshow(cross_pivot, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r",
                         title="话题×情感交叉热力图", template="plotly_dark", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)
st.divider()

# ====================== 第八行：热门话题时段分布 ======================
st.subheader("🔥 热门话题时段分布")

top_topics = df["topic"].value_counts().head(3).index.tolist()
topic_hour_data = df[df["topic"].isin(top_topics)].groupby(["topic", "hour"]).size().reset_index(name="数量")

fig_topic_hour = px.line(topic_hour_data, x="hour", y="数量", color="topic",
                         title="热门话题时段分布对比",
                         labels={"hour": "小时", "数量": "评论数"},
                         markers=True, template="plotly_dark", height=450)
fig_topic_hour.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
st.plotly_chart(fig_topic_hour, use_container_width=True)
st.divider()

# ====================== 第九行：深度分析 ======================
st.subheader("🔍 深度分析")
col_deep1, col_deep2, col_deep3 = st.columns(3)

with col_deep1:
    polar_dist = df["polar"].value_counts().reset_index()
    polar_dist.columns = ["情感倾向", "数量"]
    fig_pie = px.pie(polar_dist, values="数量", names="情感倾向", hole=0.4,
                     color="情感倾向", color_discrete_map=EMO_COLOR,
                     title="整体情感分布", template="plotly_dark", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_deep2:
    topic_simple = df["topic"].value_counts().head(6)
    fig_donut = px.pie(values=topic_simple.values, names=topic_simple.index,
                       title="话题分布占比", hole=0.3, template="plotly_dark", height=400)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_deep3:
    flow_df = df.groupby(["topic", "polar"]).size().reset_index(name="数量")
    flow_df = flow_df.head(20)
    
    all_nodes = list(flow_df["topic"].unique()) + ["强烈负面", "一般负面", "中性", "一般正面", "强烈正面"]
    topic_idx = {t: i for i, t in enumerate(flow_df["topic"].unique())}
    emo_idx = {p: len(flow_df["topic"].unique()) + i for i, p in enumerate(["强烈负面", "一般负面", "中性", "一般正面", "强烈正面"])}
    
    fig_sankey = go.Figure(go.Sankey(
        node=dict(label=all_nodes, pad=20, thickness=25, color="#4ecdc4"),
        link=dict(
            source=[topic_idx[t] for t in flow_df["topic"]],
            target=[emo_idx[p] for p in flow_df["polar"]],
            value=flow_df["数量"].tolist(),
            color="rgba(78, 205, 196, 0.4)"
        )
    ))
    fig_sankey.update_layout(title="话题→情感流向桑基图", template="plotly_dark", height=400)
    st.plotly_chart(fig_sankey, use_container_width=True)
st.divider()

# ====================== 第十行：词云 ======================
st.subheader("☁️ 评论关键词词云")

col_word1, col_word2 = st.columns(2)

with col_word1:
    try:
        text_all = " ".join(df["text"].tolist())
        font_paths = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]
        font_path = next((fp for fp in font_paths if os.path.exists(fp)), None)
        
        wc = WordCloud(background_color="#0a1128", width=800, height=450,
                       colormap="coolwarm", font_path=font_path).generate(text_all)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info("词云生成需要安装wordcloud库")

with col_word2:
    st.markdown("**负面情绪词云**")
    try:
        negative_text = " ".join(df[df["polar"].isin(["强烈负面", "一般负面"])]["text"].tolist())
        if negative_text.strip():
            wc_neg = WordCloud(background_color="#0a1128", width=800, height=450,
                               colormap="Reds", font_path=font_path).generate(negative_text)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.imshow(wc_neg, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("暂无负面评论")
    except Exception as e:
        st.info("生成失败")
st.divider()

# ====================== 第十一行：核心洞察 ======================
st.subheader("🎯 舆情核心洞察与建议")

col_insight1, col_insight2, col_insight3 = st.columns(3)

with col_insight1:
    st.markdown("**🔥 关键发现**")
    peak_hour = hour_stats.loc[hour_stats["评论数"].idxmax(), "小时"]
    lowest_sentiment_hour = hour_sentiment.loc[hour_sentiment["平均情感分"].idxmin(), "小时"]
    st.metric("评论高峰时段", f"{int(peak_hour)}:00")
    st.metric("情感最低时段", f"{int(lowest_sentiment_hour)}:00")

with col_insight2:
    st.markdown("**📊 时段特征**")
    morning_sentiment = hour_sentiment[hour_sentiment["小时"].between(6, 11)]["平均情感分"].mean()
    night_sentiment = hour_sentiment[hour_sentiment["小时"].between(18, 23)]["平均情感分"].mean()
    st.metric("早晨时段(6-11点)情感", f"{morning_sentiment:.3f}")
    st.metric("夜间时段(18-23点)情感", f"{night_sentiment:.3f}", delta=f"{night_sentiment - morning_sentiment:.3f}")

with col_insight3:
    st.markdown("**💡 治理建议**")
    negative_hours = df[df["polar"].isin(["强烈负面", "一般负面"])].groupby("hour").size()
    if len(negative_hours) > 0:
        peak_negative_hour = negative_hours.idxmax()
        st.warning(f"⚠️ 负面评论高峰时段：{int(peak_negative_hour)}:00")
        st.info(f"建议在该时段加强舆情监控和快速响应机制")

st.divider()

# ====================== 临时分析结果 ======================
if st.session_state.temp_result:
    st.subheader("📋 最新评论分析结果")
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    with col_res1:
        st.metric("评论内容", st.session_state.temp_result["评论内容"])
    with col_res2:
        st.metric("话题分类", st.session_state.temp_result["话题分类"])
    with col_res3:
        st.metric("情感评分", st.session_state.temp_result["情感评分"])
    with col_res4:
        st.metric("情感倾向", st.session_state.temp_result["情感倾向"])

# ====================== 数据预览 ======================
with st.expander("📊 查看原始数据预览"):
    st.dataframe(df[["time", "text", "area", "hour", "weekday_name", "topic", "polar", "sentiment"]].head(100), 
                 use_container_width=True)
    st.caption(f"共 {len(df)} 条评论数据")
