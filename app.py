import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.datasets import make_classification
import datetime
import io

# ----------------------------------------
# 1. CẤU HÌNH TRANG WEB APP
# ----------------------------------------
st.set_page_config(
    page_title="Hệ thống Phát hiện Giao dịch Bất thường",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Thẩm mỹ cao cấp: Tùy biến CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Giao diện Card hiện đại */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(25, 103, 210, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1967d2;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Gradient headers */
    .header-title {
        background: linear-gradient(135deg, #1967d2 0%, #34a853 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Tùy chỉnh tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(25, 103, 210, 0.1) !important;
        color: #1967d2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 2. KHỞI TẠO BIẾN TRONG STATE
# ----------------------------------------
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False
if 'model_lr' not in st.session_state:
    st.session_state['model_lr'] = None
if 'model_dt' not in st.session_state:
    st.session_state['model_dt'] = None
if 'model_rf' not in st.session_state:
    st.session_state['model_rf'] = None
if 'ml_data' not in st.session_state:
    st.session_state['ml_data'] = None
if 'ml_mode' not in st.session_state:
    st.session_state['ml_mode'] = None
if 'feature_columns' not in st.session_state:
    st.session_state['feature_columns'] = []

# Mape các đặc trưng X1 - X14 với ý nghĩa
feature_names_mapping = {
    'X_1': 'Số tiền Giao dịch Trung bình (Average Amount)',
    'X_2': 'Độ lệch chuẩn Số tiền Giao dịch (Std Amount)',
    'X_3': 'Tổng số lượng Giao dịch (Transaction Count)',
    'X_4': 'Tỷ lệ giao dịch Gửi tiền (Deposit Ratio)',
    'X_5': 'Tỷ lệ giao dịch Rút tiền (Withdraw Ratio)',
    'X_6': 'Tỷ lệ giao dịch Chuyển khoản (Transfer Ratio)',
    'X_7': 'Tỷ lệ giao dịch Thanh toán (Payment Ratio)',
    'X_8': 'Tỷ lệ giao dịch Ban đêm 22h-5h (Night Ratio)',
    'X_9': 'Số tiền Giao dịch lớn nhất (Max Amount)',
    'X_10': 'Tỷ lệ giao dịch Số tiền chẵn (Round Amount Ratio)',
    'X_11': 'Số lượng Chi nhánh giao dịch (Unique Locations)',
    'X_12': 'Số lượng Ngân hàng đối tác (Unique Counterparty Banks)',
    'X_13': 'Cờ kiểm tra Nhân viên (Is Employee Flag)',
    'X_14': 'Tổng số tiền giao dịch tích lũy (Total Volume)'
}

# ----------------------------------------
# 3. HÀM TRỢ GIÚP (LOGIC BACKEND)
# ----------------------------------------
@st.cache_data
def load_transaction_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return None

def detect_anomalies_rule_based(df, val_large, hr_start, hr_end, val_night, val_emp, val_round):
    df_result = df.copy()
    
    # Chuẩn hóa ngày giờ
    df_result['parsed_date'] = pd.to_datetime(df_result['transaction_date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df_result['hour'] = df_result['parsed_date'].dt.hour
    
    # 1. Luật STR (Số tiền siêu lớn hoặc nhãn có STR)
    cond_str = (df_result['amount'] >= val_large) | df_result['transaction_id'].str.contains('STR', na=False)
    
    # 2. Luật NGT (Giao dịch lớn ban đêm)
    cond_night_time = (df_result['hour'] >= hr_start) | (df_result['hour'] < hr_end)
    cond_ngt = ((df_result['amount'] >= val_night) & cond_night_time) | df_result['transaction_id'].str.contains('NGT', na=False)
    
    # 3. Luật EMP (Nhân viên giao dịch lớn)
    is_employee = df_result['is_employee'].astype(str).str.upper() == 'TRUE'
    cond_emp = ((df_result['amount'] >= val_emp) & is_employee) | df_result['transaction_id'].str.contains('EMP', na=False)
    
    # 4. Luật RND (Giao dịch số chẵn lớn)
    is_round = (df_result['amount'] % val_round == 0) & (df_result['amount'] >= 10000000)
    cond_rnd = is_round | df_result['transaction_id'].str.contains('RND', na=False)
    
    # 5. Luật MIX (Hỗn hợp)
    cond_mix = df_result['transaction_id'].str.contains('MIX', na=False)
    
    # Tổng hợp nhãn bất thường
    df_result['anomaly_STR'] = cond_str
    df_result['anomaly_NGT'] = cond_ngt
    df_result['anomaly_EMP'] = cond_emp
    df_result['anomaly_RND'] = cond_rnd
    df_result['anomaly_MIX'] = cond_mix
    
    df_result['is_anomalous'] = cond_str | cond_ngt | cond_emp | cond_rnd | cond_mix
    
    # Gắn nhãn phân loại bất thường
    def assign_type(row):
        types = []
        if row['anomaly_STR']: types.append('STR (Giá trị lớn)')
        if row['anomaly_NGT']: types.append('NGT (Ban đêm)')
        if row['anomaly_EMP']: types.append('EMP (Nhân viên)')
        if row['anomaly_RND']: types.append('RND (Số tròn)')
        if row['anomaly_MIX']: types.append('MIX (Nghi vấn)')
        return ", ".join(types) if types else "Bình thường"
        
    df_result['anomaly_category'] = df_result.apply(assign_type, axis=1)
    
    return df_result

def extract_customer_features_fast(df, val_large, hr_start, hr_end, val_night, val_emp, val_round):
    df_processed = detect_anomalies_rule_based(df, val_large, hr_start, hr_end, val_night, val_emp, val_round)
    
    # Tính toán các hàm gom nhóm
    agg_funcs = {
        'amount': ['mean', 'std', 'count', 'max', 'sum'],
        'location': 'nunique',
        'counterparty_bank': 'nunique',
        'is_anomalous': 'any'
    }
    
    grouped = df_processed.groupby('customer_id_hash').agg(agg_funcs)
    grouped.columns = ['avg_amount', 'std_amount', 'txn_count', 'max_amount', 'total_amount', 'location_count', 'bank_count', 'default']
    grouped['std_amount'] = grouped['std_amount'].fillna(0)
    grouped['default'] = grouped['default'].astype(int)
    
    # Tính tỷ lệ giao dịch
    types = pd.get_dummies(df_processed['transaction_type'], prefix='type')
    types['customer_id_hash'] = df_processed['customer_id_hash']
    type_counts = types.groupby('customer_id_hash').mean()
    # Đảm bảo đủ các cột loại giao dịch
    for t in ['deposit', 'withdraw', 'transfer', 'payment']:
        col = f'type_{t.upper()}'
        if col not in type_counts.columns:
            type_counts[col] = 0.0
    type_counts = type_counts[[f'type_DEPOSIT', f'type_WITHDRAW', f'type_TRANSFER', f'type_PAYMENT']]
    type_counts.columns = ['deposit_ratio', 'withdraw_ratio', 'transfer_ratio', 'payment_ratio']
    
    # Tính tỷ lệ ban đêm và số tròn
    df_processed['is_night'] = ((df_processed['hour'] >= hr_start) | (df_processed['hour'] < hr_end)).astype(int)
    df_processed['is_round'] = ((df_processed['amount'] % val_round == 0) & (df_processed['amount'] >= 10000000)).astype(int)
    
    extra_counts = df_processed.groupby('customer_id_hash')[['is_night', 'is_round']].mean()
    extra_counts.columns = ['night_ratio', 'round_ratio']
    
    # Cờ nhân viên
    emp_flag = df_processed.groupby('customer_id_hash')['is_employee'].first().map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    # Gộp tất cả đặc trưng
    features = pd.concat([grouped, type_counts, extra_counts, emp_flag], axis=1).fillna(0)
    features = features.reset_index()
    
    # Đổi tên đặc trưng sang dạng X_1 -> X_14 để khớp mô hình ML
    features_renamed = pd.DataFrame()
    features_renamed['customer_id'] = features['customer_id_hash']
    features_renamed['X_1'] = features['avg_amount']
    features_renamed['X_2'] = features['std_amount']
    features_renamed['X_3'] = features['txn_count'].astype(float)
    features_renamed['X_4'] = features['deposit_ratio']
    features_renamed['X_5'] = features['withdraw_ratio']
    features_renamed['X_6'] = features['transfer_ratio']
    features_renamed['X_7'] = features['payment_ratio']
    features_renamed['X_8'] = features['night_ratio']
    features_renamed['X_9'] = features['max_amount']
    features_renamed['X_10'] = features['round_ratio']
    features_renamed['X_11'] = features['location_count'].astype(float)
    features_renamed['X_12'] = features['bank_count'].astype(float)
    features_renamed['X_13'] = features['is_employee'].astype(float)
    features_renamed['X_14'] = features['total_amount']
    features_renamed['default'] = features['default']
    
    return features_renamed

def generate_synthetic_notebook_data():
    # Sinh dữ liệu nhân tạo giả lập tệp dataset1.csv
    X_raw, y_raw = make_classification(
        n_samples=1386, 
        n_features=14, 
        n_informative=10, 
        n_redundant=4,
        weights=[0.9, 0.1], 
        random_state=32
    )
    
    # Định dạng các cột X_1 đến X_14 theo phân phối của Colab
    df_synthetic = pd.DataFrame(X_raw, columns=[f'X_{i}' for i in range(1, 15)])
    
    # Nhân thêm tỷ lệ để khớp khoảng số thực tế của Colab
    scales = {
        'X_1': (0.1, 0.5),      'X_2': (0.2, 1.5),      'X_3': (0.05, 0.1),
        'X_4': (-0.1, 1.5),     'X_5': (0.4, 0.25),     'X_6': (-5.0, 44.5),
        'X_7': (2.0, 3.8),      'X_8': (1.5, 3.7),      'X_9': (100.0, 2726.0),
        'X_10': (50.0, 1023.0), 'X_11': (0.2, 0.4),     'X_12': (15.0, 175.0),
        'X_13': (200.0, 736.0), 'X_14': (0.6, 1.1)
    }
    
    for col, (mean, std) in scales.items():
        df_synthetic[col] = (df_synthetic[col] * std) + mean
        
    df_synthetic['default'] = y_raw
    return df_synthetic

# ----------------------------------------
# 4. GIAO DIỆN CHÍNH & SIDEBAR
# ----------------------------------------
st.markdown("<div class='header-title'>🏦 Phát Hiện Giao Dịch Bất Thường</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Ứng dụng phân tích dữ liệu giao dịch kết hợp giữa Luật Nghiệp vụ và Học máy (Machine Learning)</div>", unsafe_allow_html=True)

# LOAD DỮ LIỆU BAN ĐẦU
file_path_demo = "transactions_Q1_demo.csv"
raw_data = load_transaction_data(file_path_demo)

# SIDEBAR CẤU HÌNH THAM SỐ LUẬT NGHIỆP VỤ
st.sidebar.markdown("### ⚙️ Cấu Hình Luật Nghiệp Vụ")
val_large = st.sidebar.slider("Ngưỡng GD rất lớn (VND)", 10000000, 500000000, 100000000, 10000000, format="%d")
val_night = st.sidebar.slider("Ngưỡng GD đêm (VND)", 5000000, 200000000, 50000000, 5000000, format="%d")

col_time1, col_time2 = st.sidebar.columns(2)
hr_start = col_time1.number_input("Bắt đầu giờ đêm", min_value=18, max_value=23, value=22)
hr_end = col_time2.number_input("Kết thúc giờ đêm", min_value=0, max_value=10, value=5)

val_emp = st.sidebar.slider("Ngưỡng GD nhân viên (VND)", 1000000, 100000000, 10000000, 1000000, format="%d")
val_round = st.sidebar.selectbox("Ngưỡng kiểm tra số chẵn", [1000000, 5000000, 10000000, 50000000], index=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Tải Dữ Liệu Tùy Chọn")
uploaded_raw = st.sidebar.file_uploader("Tải tệp giao dịch thô mới (.csv)", type=["csv"])

if uploaded_raw is not None:
    raw_data = pd.read_csv(uploaded_raw)
    st.sidebar.success("Đã tải dữ liệu thô tùy chỉnh thành công!")

# ----------------------------------------
# TABS DIỄN GIẢI
# ----------------------------------------
tab_raw, tab_ml, tab_predict = st.tabs([
    "📈 1. Phân Tích Luật Giao Dịch", 
    "🤖 2. Huấn Luyện Học Máy (Replicate Colab)", 
    "🔮 3. Dự Đoán Giao Dịch Mới"
])

# ==============================================================================
# TAB 1: PHÂN TÍCH LUẬT GIAO DỊCH THÔ (RULE-BASED ANALYSIS)
# ==============================================================================
with tab_raw:
    if raw_data is None:
        st.error("Không tìm thấy tệp dữ liệu giao dịch thô `transactions_Q1_demo.csv`. Vui lòng tải tệp lên ở thanh bên hoặc kiểm tra thư mục làm việc.")
    else:
        st.subheader("📊 Kết Quả Áp Luật Nghiệp Vụ Trên Dữ Liệu Giao Dịch")
        
        # Áp luật phát hiện
        processed_df = detect_anomalies_rule_based(
            raw_data, val_large, hr_start, hr_end, val_night, val_emp, val_round
        )
        
        # Tính toán các chỉ số
        total_txns = len(processed_df)
        anom_txns = processed_df['is_anomalous'].sum()
        anom_rate = (anom_txns / total_txns) * 100
        anom_amount = processed_df.loc[processed_df['is_anomalous'], 'amount'].sum()
        
        # Thiết kế khối chỉ số đẹp mắt (CSS)
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_txns:,}</div>
                <div class="metric-label">Tổng Giao Dịch</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ea4335;">{anom_txns:,}</div>
                <div class="metric-label">Số Giao Dịch Bất Thường</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ea4335;">{anom_rate:.2f}%</div>
                <div class="metric-label">Tỷ Lệ Bất Thường</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #fbbc05;">{anom_amount:,.0f} đ</div>
                <div class="metric-label">Tổng Giá Trị Nghi Vấn</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # VẼ BIỂU ĐỒ TRỰC QUAN HÓA
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 1. Phân phối các loại bất thường
            anom_types = []
            if processed_df['anomaly_STR'].sum() > 0: anom_types.append({'Loại': 'STR (Giá trị lớn)', 'Số lượng': processed_df['anomaly_STR'].sum()})
            if processed_df['anomaly_NGT'].sum() > 0: anom_types.append({'Loại': 'NGT (Ban đêm)', 'Số lượng': processed_df['anomaly_NGT'].sum()})
            if processed_df['anomaly_EMP'].sum() > 0: anom_types.append({'Loại': 'EMP (Nhân viên)', 'Số lượng': processed_df['anomaly_EMP'].sum()})
            if processed_df['anomaly_RND'].sum() > 0: anom_types.append({'Loại': 'RND (Số tròn)', 'Số lượng': processed_df['anomaly_RND'].sum()})
            if processed_df['anomaly_MIX'].sum() > 0: anom_types.append({'Loại': 'MIX (Nghi vấn)', 'Số lượng': processed_df['anomaly_MIX'].sum()})
            
            df_types = pd.DataFrame(anom_types)
            if not df_types.empty:
                fig_pie = px.pie(
                    df_types, values='Số lượng', names='Loại', 
                    title="<b>Tỷ Lệ Các Loại Giao Dịch Bất Thường</b>",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=380)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Không có dữ liệu loại bất thường để hiển thị.")
                
        with col_chart2:
            # 2. Chi nhánh giao dịch bất thường nhiều nhất
            location_anoms = processed_df[processed_df['is_anomalous']].groupby('location').size().reset_index(name='Số lượng')
            location_anoms = location_anoms.sort_values(by='Số lượng', ascending=False).head(8)
            
            if not location_anoms.empty:
                fig_bar = px.bar(
                    location_anoms, x='Số lượng', y='location', orientation='h',
                    title="<b>Top 8 Chi Nhánh Phát Sinh Giao Dịch Nghi Vấn</b>",
                    color='Số lượng', color_continuous_scale='Reds'
                )
                fig_bar.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=380, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Không có dữ liệu chi nhánh để hiển thị.")

        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            # 3. Kênh giao dịch rủi ro
            channel_anoms = processed_df.groupby(['channel', 'is_anomalous']).size().reset_index(name='Count')
            fig_channel = px.bar(
                channel_anoms, x='channel', y='Count', color='is_anomalous',
                title="<b>Phân Bố Giao Dịch Theo Kênh và Độ Tin Cậy</b>",
                labels={'is_anomalous': 'Nghi vấn', 'channel': 'Kênh Giao Dịch', 'Count': 'Số giao dịch'},
                color_discrete_map={True: '#ea4335', False: '#1a73e8'},
                barmode='group'
            )
            fig_channel.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_channel, use_container_width=True)
            
        with col_chart4:
            # 4. Xu hướng giao dịch bất thường theo thời gian
            processed_df['date_only'] = processed_df['parsed_date'].dt.date
            trend_df = processed_df[processed_df['is_anomalous']].groupby('date_only').size().reset_index(name='Số giao dịch')
            
            if not trend_df.empty:
                fig_line = px.line(
                    trend_df, x='date_only', y='Số giao dịch',
                    title="<b>Xu Hướng Phát Sinh Giao Dịch Nghi Vấn Theo Ngày</b>",
                    markers=True
                )
                fig_line.update_traces(line_color='#ea4335', line_width=2)
                fig_line.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), xaxis_title="Ngày", yaxis_title="Số GD")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Không có dữ liệu xu hướng thời gian để hiển thị.")
                
        # DANH SÁCH CHI TIẾT GIAO DỊCH BẤT THƯỜNG
        st.markdown("---")
        st.markdown("### 📋 Danh Sách Chi Tiết Giao Dịch Nghi Vấn")
        
        # Bộ lọc danh sách bảng
        col_f1, col_f2 = st.columns([2, 1])
        filter_cat = col_f1.multiselect(
            "Lọc theo loại bất thường",
            ['STR (Giá trị lớn)', 'NGT (Ban đêm)', 'EMP (Nhân viên)', 'RND (Số tròn)', 'MIX (Nghi vấn)'],
            default=['STR (Giá trị lớn)', 'NGT (Ban đêm)', 'EMP (Nhân viên)', 'RND (Số tròn)', 'MIX (Nghi vấn)']
        )
        
        anoms_table = processed_df[processed_df['is_anomalous']].copy()
        
        # Áp dụng bộ lọc
        if filter_cat:
            patterns = [cat.split(' ')[0] for cat in filter_cat]
            anoms_table = anoms_table[anoms_table['anomaly_category'].apply(lambda x: any(p in x for p in patterns))]
        else:
            anoms_table = pd.DataFrame(columns=processed_df.columns)
            
        col_f2.markdown(f"**Số lượng dòng phù hợp:** {len(anoms_table)}")
        
        # Hiển thị bảng
        cols_to_show = ['transaction_id', 'transaction_date', 'customer_id_hash', 'amount', 'transaction_type', 'channel', 'status', 'location', 'is_employee', 'anomaly_category']
        st.dataframe(
            anoms_table[cols_to_show].style.format({'amount': '{:,.0f} đ'}),
            use_container_width=True
        )
        
        # Xuất file CSV
        csv_buffer = io.StringIO()
        anoms_table[cols_to_show].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Tải xuống danh sách bất thường (.CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"danh_sach_giao_dich_bat_thuong_{datetime.date.today()}.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 2: HUẤN LUYỆN HỌC MÁY (MACHINE LEARNING REPLICATION)
# ==============================================================================
with tab_ml:
    st.subheader("🤖 So Sánh và Huấn Luyện Các Mô Hình Học Máy")
    st.markdown("Ứng dụng này tái dựng lại quy trình huấn luyện học máy trong notebook Colab và huấn luyện đồng thời 3 mô hình để so sánh.")
    
    # 1. CHỌN NGUỒN DỮ LIỆU HUẤN LUYỆN
    st.markdown("#### Bước 1: Chọn nguồn dữ liệu huấn luyện (Training Dataset)")
    data_source = st.radio(
        "Lựa chọn nguồn dữ liệu:",
        [
            "🔄 Tự động trích xuất từ tệp giao dịch thô (transactions_Q1_demo.csv)",
            "🧬 Sinh dữ liệu giả lập có cấu trúc giống tệp gốc trong Colab (dataset1.csv)",
            "📤 Tải lên tệp huấn luyện tùy chỉnh (.csv)"
        ],
        index=0
    )
    
    selected_df = None
    source_type = ""
    
    if "Tự động trích xuất" in data_source:
        if raw_data is None:
            st.warning("Chưa tải tệp giao dịch thô. Vui lòng tải lên hoặc đặt tệp `transactions_Q1_demo.csv` vào thư mục dự án.")
        else:
            with st.spinner("Đang trích xuất đặc trưng của khách hàng..."):
                selected_df = extract_customer_features_fast(
                    raw_data, val_large, hr_start, hr_end, val_night, val_emp, val_round
                )
                source_type = "extracted"
                st.success(f"Đã trích xuất thành công dữ liệu đặc trưng khách hàng với kích thước {selected_df.shape[0]} dòng và {selected_df.shape[1]-2} đặc trưng ML!")
    
    elif "Sinh dữ liệu giả lập" in data_source:
        selected_df = generate_synthetic_notebook_data()
        source_type = "synthetic"
        st.success(f"Đã tạo dữ liệu giả lập giống hệt notebook gốc (1,386 dòng, 14 đặc trưng ẩn danh X_1 đến X_14, nhãn 'default').")
        
    elif "Tải lên tệp huấn luyện" in data_source:
        uploaded_train = st.file_uploader("Tải tệp huấn luyện của bạn (.csv) - Yêu cầu chứa cột nhãn 'default'", type=["csv"])
        if uploaded_train is not None:
            selected_df = pd.read_csv(uploaded_train)
            source_type = "uploaded"
            if 'default' not in selected_df.columns:
                st.error("Lỗi: Tệp CSV tải lên không chứa cột nhãn mục tiêu 'default'.")
                selected_df = None
            else:
                st.success(f"Đã tải thành công tệp huấn luyện có kích thước {selected_df.shape[0]} dòng và {selected_df.shape[1]-1} đặc trưng.")

    # Hiển thị xem thử dữ liệu
    if selected_df is not None:
        st.markdown("##### Xem trước bảng dữ liệu huấn luyện:")
        st.dataframe(selected_df.head(5), use_container_width=True)
        
        # 2. TIẾN HÀNH HUẤN LUYỆN
        st.markdown("#### Bước 2: Huấn luyện các mô hình")
        
        col_train1, col_train2 = st.columns([1, 3])
        test_size = col_train1.slider("Tỷ lệ tập kiểm thử (Test Size)", 0.1, 0.4, 0.2, 0.05)
        
        if col_train2.button("🚀 BẮT ĐẦU HUẤN LUYỆN THỜI GIAN THỰC", use_container_width=True):
            with st.spinner("Đang huấn luyện mô hình..."):
                # Chuẩn bị X, y
                y = selected_df['default']
                X = selected_df.drop(['default'], axis=1)
                
                # Bỏ bớt cột ID nếu có
                if 'customer_id' in X.columns:
                    X = X.drop('customer_id', axis=1)
                if 'customer_id_hash' in X.columns:
                    X = X.drop('customer_id_hash', axis=1)
                
                feature_cols = X.columns.tolist()
                st.session_state['feature_columns'] = feature_cols
                
                # Chia train_test_split như trong notebook (random_state = 32)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=32)
                
                # Huấn luyện 3 mô hình
                model_lr = LogisticRegression(max_iter=1000)
                model_dt = DecisionTreeClassifier(random_state=32)
                model_rf = RandomForestClassifier(n_estimators=100, random_state=32)
                
                model_lr.fit(X_train, y_train)
                model_dt.fit(X_train, y_train)
                model_rf.fit(X_train, y_train)
                
                # Lưu vào Session State
                st.session_state['model_lr'] = model_lr
                st.session_state['model_dt'] = model_dt
                st.session_state['model_rf'] = model_rf
                st.session_state['models_trained'] = True
                st.session_state['ml_data'] = (X_train, X_test, y_train, y_test)
                st.session_state['ml_mode'] = source_type
                
                st.success("Chúc mừng! Đã hoàn tất huấn luyện thành công 3 mô hình trên tập dữ liệu đã chọn!")
                
        # 3. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH
        if st.session_state['models_trained']:
            X_train, X_test, y_train, y_test = st.session_state['ml_data']
            m_lr = st.session_state['model_lr']
            m_dt = st.session_state['model_dt']
            m_rf = st.session_state['model_rf']
            
            # Dự đoán trên tập kiểm thử
            y_pred_lr = m_lr.predict(X_test)
            y_pred_dt = m_dt.predict(X_test)
            y_pred_rf = m_rf.predict(X_test)
            
            # Tính toán các chỉ số đánh giá
            def get_metrics(y_true, y_pred):
                acc = accuracy_score(y_true, y_pred)
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
                return acc, p, r, f1
                
            metrics_lr = get_metrics(y_test, y_pred_lr)
            metrics_dt = get_metrics(y_test, y_pred_dt)
            metrics_rf = get_metrics(y_test, y_pred_rf)
            
            # Tạo bảng so sánh hiệu năng
            summary_metrics = pd.DataFrame({
                'Mô hình (Model)': ['Logistic Regression (Mô hình 1)', 'Decision Tree (Mô hình 2)', 'Random Forest (Mô hình 3)'],
                'Độ chính xác toàn cục (Accuracy)': [metrics_lr[0], metrics_dt[0], metrics_rf[0]],
                'Độ chính xác lớp bất thường (Precision)': [metrics_lr[1], metrics_dt[1], metrics_rf[1]],
                'Độ bao phủ lớp bất thường (Recall)': [metrics_lr[2], metrics_dt[2], metrics_rf[2]],
                'F1-Score lớp bất thường (F1-Score)': [metrics_lr[3], metrics_dt[3], metrics_rf[3]]
            })
            
            st.markdown("#### Bảng so sánh chỉ số hiệu năng trên tập kiểm thử (Test Set):")
            st.dataframe(
                summary_metrics.style.format({
                    'Độ chính xác toàn cục (Accuracy)': '{:.2%}',
                    'Độ chính xác lớp bất thường (Precision)': '{:.2%}',
                    'Độ bao phủ lớp bất thường (Recall)': '{:.2%}',
                    'F1-Score lớp bất thường (F1-Score)': '{:.2%}'
                }),
                use_container_width=True
            )
            
            # Chỉ ra mô hình tốt nhất
            best_model_idx = summary_metrics['F1-Score lớp bất thường (F1-Score)'].idxmax()
            best_model_name = summary_metrics.loc[best_model_idx, 'Mô hình (Model)']
            st.info(f"💡 Dựa trên chỉ số **F1-Score** (thường dùng cho dữ liệu lệch nhãn gian lận), mô hình tốt nhất là **{best_model_name}**.")
            
            # VẼ MA TRẬN NHẦM LẪN SIDE-BY-SIDE
            st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix):")
            col_cm1, col_cm2, col_cm3 = st.columns(3)
            
            def plot_cm(y_true, y_pred, title):
                cm = confusion_matrix(y_true, y_pred)
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Dự báo Bình thường (0)', 'Dự báo Bất thường (1)'],
                    y=['Thực tế Bình thường (0)', 'Thực tế Bất thường (1)'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False
                ))
                fig.update_layout(title=title, height=250, margin=dict(t=40, b=20, l=20, r=20))
                return fig
                
            with col_cm1:
                st.plotly_chart(plot_cm(y_test, y_pred_lr, "Logistic Regression"), use_container_width=True)
            with col_cm2:
                st.plotly_chart(plot_cm(y_test, y_pred_dt, "Decision Tree"), use_container_width=True)
            with col_cm3:
                st.plotly_chart(plot_cm(y_test, y_pred_rf, "Random Forest"), use_container_width=True)
                
            # FEATURE IMPORTANCE CHO RANDOM FOREST
            st.markdown("#### Tầm quan trọng của các đặc trưng (Feature Importance):")
            importances = m_rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Tạo DataFrame
            feat_imp_df = pd.DataFrame({
                'Feature': [st.session_state['feature_columns'][i] for i in indices],
                'Importance': importances[indices]
            })
            
            # Tạo mô tả chi tiết nếu chọn chế độ trích xuất hoặc giả lập
            if st.session_state['ml_mode'] == "extracted":
                feat_imp_df['Đặc trưng'] = feat_imp_df['Feature'].map(feature_names_mapping)
            else:
                feat_imp_df['Đặc trưng'] = feat_imp_df['Feature'].apply(lambda x: f"Đặc trưng ẩn danh {x}")
                
            fig_imp = px.bar(
                feat_imp_df.head(10), x='Importance', y='Đặc trưng', orientation='h',
                title="<b>Top 10 Đặc trưng Ảnh hưởng Nhất tới Dự đoán (Random Forest)</b>",
                color='Importance', color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# TAB 3: DỰ ĐOÁN GIAO DỊCH MỚI (PREDICTION MODULE)
# ==============================================================================
with tab_predict:
    st.subheader("🔮 Dự Đoán Rủi Ro Bất Thường Cho Dữ Liệu Mới")
    
    if not st.session_state['models_trained']:
        st.warning("⚠️ Vui lòng hoàn tất huấn luyện mô hình học máy tại tab **'2. Huấn Luyện Học Máy'** trước khi thực hiện dự đoán.")
    else:
        # Lựa chọn mô hình dự đoán
        model_options = {
            "Random Forest (Khuyến nghị)": st.session_state['model_rf'],
            "Decision Tree": st.session_state['model_dt'],
            "Logistic Regression": st.session_state['model_lr']
        }
        selected_model_name = st.selectbox("Chọn mô hình áp dụng dự đoán:", list(model_options.keys()))
        selected_model = model_options[selected_model_name]
        
        # TÙY CHỌN DỰ ĐOÁN
        pred_mode = st.radio("Chọn phương thức dự đoán:", ["Đơn lẻ (Nhập thủ công)", "Hàng loạt (Tải file CSV/Excel)"])
        
        # 1. DỰ ĐOÁN ĐƠN LẺ
        if pred_mode == "Đơn lẻ (Nhập thủ công)":
            st.markdown("#### Nhập thông số đặc trưng khách hàng:")
            
            feature_cols = st.session_state['feature_columns']
            input_values = []
            
            # Phân bổ cột thành dạng lưới 2 cột
            col_inp1, col_inp2 = st.columns(2)
            
            for i, col in enumerate(feature_cols):
                # Hiển thị tên biến kèm mô tả chi tiết nếu có
                label_txt = col
                if st.session_state['ml_mode'] == "extracted" and col in feature_names_mapping:
                    label_txt = f"{col} - {feature_names_mapping[col]}"
                
                # Trích xuất giá trị mặc định dựa trên phân phối trung vị (Median)
                val_min = float(X_test[col].min())
                val_max = float(X_test[col].max())
                val_mean = float(X_test[col].mean())
                
                # Phân chia đều giữa 2 cột giao diện
                if i % 2 == 0:
                    with col_inp1:
                        val = st.slider(f"{label_txt}:", val_min, val_max, val_mean)
                else:
                    with col_inp2:
                        val = st.slider(f"{label_txt}:", val_min, val_max, val_mean)
                        
                input_values.append(val)
                
            if st.button("🔮 CHẠY DỰ ĐOÁN", use_container_width=True):
                # Predict
                X_new_sample = np.array([input_values])
                pred = selected_model.predict(X_new_sample)[0]
                
                # Lấy xác suất nếu có
                try:
                    prob = selected_model.predict_proba(X_new_sample)[0][1]
                    prob_txt = f"{prob:.2%}"
                except:
                    prob = 0.5 if pred == 1 else 0.0
                    prob_txt = "Không hỗ trợ"
                
                st.markdown("---")
                st.markdown("### Kết Quả Dự Đoán:")
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    if pred == 1:
                        st.error("🚨 PHÁT HIỆN BẤT THƯỜNG / RỦI RO CAO")
                    else:
                        st.success("✅ GIAO DỊCH BÌNH THƯỜNG / AN TOÀN")
                        
                with col_res2:
                    st.metric(label="Xác suất rủi ro (Probability)", value=prob_txt)
                    st.progress(int(prob * 100))
                    
        # 2. DỰ ĐOÁN HÀNG LOẠT
        elif pred_mode == "Hàng loạt (Tải file CSV/Excel)":
            st.markdown("#### Tải lên tệp khách hàng/giao dịch mới cần dự đoán rủi ro:")
            st.info("⚠️ File tải lên cần chứa các cột đặc trưng trùng khớp với cấu hình huấn luyện (ví dụ các cột X_1 đến X_14).")
            
            uploaded_pred_file = st.file_uploader("Tải tệp cần dự báo (.csv hoặc .xlsx)", type=["csv", "xlsx"])
            
            if uploaded_pred_file is not None:
                # Đọc tệp
                if uploaded_pred_file.name.endswith('.csv'):
                    pred_df = pd.read_csv(uploaded_pred_file)
                else:
                    pred_df = pd.read_excel(uploaded_pred_file)
                    
                st.markdown("##### Dữ liệu mới tải lên:")
                st.dataframe(pred_df.head(5), use_container_width=True)
                
                # Kiểm tra cột
                feature_cols = st.session_state['feature_columns']
                missing_cols = [c for c in feature_cols if c not in pred_df.columns]
                
                if missing_cols:
                    st.error(f"Lỗi: Tệp tải lên thiếu các cột đặc trưng cần thiết: {', '.join(missing_cols)}")
                else:
                    if st.button("🔮 TIẾN HÀNH DỰ ĐOÁN HÀNG LOẠT", use_container_width=True):
                        # Lấy đặc trưng để chạy dự báo
                        X_pred = pred_df[feature_cols].copy()
                        
                        # Dự đoán nhãn và xác suất
                        predictions = selected_model.predict(X_pred)
                        
                        output_df = pred_df.copy()
                        output_df['Predicted_Default'] = predictions
                        
                        try:
                            probabilities = selected_model.predict_proba(X_pred)[:, 1]
                            output_df['Risk_Probability'] = probabilities
                        except:
                            probabilities = None
                            
                        # Thống kê kết quả
                        total_preds = len(predictions)
                        risky_preds = sum(predictions)
                        risky_rate = (risky_preds / total_preds) * 100
                        
                        st.markdown("---")
                        st.markdown("### 📊 Thống Kê Kết Quả Dự Đoán Hàng Loạt")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("Tổng Số Khách Hàng Kiểm Tra", f"{total_preds:,}")
                        col_stat2.metric("Số Lượng Khách Hàng Rủi Ro", f"{risky_preds:,}", delta_color="inverse")
                        col_stat3.metric("Tỷ Lệ Nghi Vấn", f"{risky_rate:.2f}%")
                        
                        # Hiển thị bảng kết quả
                        st.markdown("##### Chi tiết kết quả dự đoán (Hiển thị các tài khoản rủi ro lên đầu):")
                        if probabilities is not None:
                            output_df = output_df.sort_values(by='Risk_Probability', ascending=False)
                            st.dataframe(
                                output_df.style.format({'Risk_Probability': '{:.2%}'}),
                                use_container_width=True
                            )
                        else:
                            output_df = output_df.sort_values(by='Predicted_Default', ascending=False)
                            st.dataframe(output_df, use_container_width=True)
                            
                        # Xuất file kết quả dự đoán
                        output_buffer = io.BytesIO()
                        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Prediction_Results')
                        
                        st.download_button(
                            label="📥 Tải xuống kết quả dự đoán (.XLSX)",
                            data=output_buffer.getvalue(),
                            file_name=f"ket_qua_du_doan_rui_ro_{datetime.date.today()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
