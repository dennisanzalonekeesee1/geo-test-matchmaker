import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

st.set_page_config(page_title="Geo-Test Matchmaker", layout="wide")

st.title("📍 Geo-Test Matchmaker & Randomizer")
st.markdown("Automate your DMA trimming, daily/weekly correlation waterfall, and randomization.")

# --- SIDEBAR: UPLOADS & SETTINGS ---
with st.sidebar:
    st.header("1. Upload Data")
    sales_file = st.file_uploader("Upload Shopify Sales", type=["csv", "xlsx"])
    zip_dma_file = st.file_uploader("Upload Zip-to-DMA Dict", type=["csv", "xlsx"])
    
    st.header("2. Settings")
    min_corr = st.slider("Target Correlation Threshold", 0.70, 0.99, 0.85, 0.01)
    
    st.markdown("### Verify Column Names")
    # Updated default names to match your files automatically!
    date_col = st.text_input("Date Column (Sales)", "Day")
    zip_col = st.text_input("Zip Code Column (Sales)", "Shipping postal code")
    sales_col = st.text_input("Sales Column (Sales)", "Gross sales")
    dma_col = st.text_input("DMA Column (Dictionary)", "dma_description")
    dict_zip_col = st.text_input("Zip Column (Dictionary)", "zip_code")

# --- CACHED FUNCTIONS FOR SPEED & STABILITY ---
# @st.cache_data tells the app to "memorize" the math so it doesn't re-flip the coins!

@st.cache_data
def load_data(sales_file, zip_dma_file):
    df_sales = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
    df_map = pd.read_csv(zip_dma_file) if zip_dma_file.name.endswith('.csv') else pd.read_excel(zip_dma_file)
    return df_sales, df_map

@st.cache_data
def process_data(df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr):
    # Make copies to prevent altering the cached original data
    df_sales = df_sales_raw.copy()
    df_map = df_map_raw.copy()
    
    # Pre-Processing
    df_sales[date_col] = pd.to_datetime(df_sales[date_col])
    df_sales[sales_col] = pd.to_numeric(df_sales[sales_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
    df_sales = df_sales[df_sales[sales_col] > 0] 
    
    # Clean Zip Codes 
    df_sales['Clean_Zip'] = df_sales[zip_col].astype(str).str.extract(r'(\d{5})')[0]
    df_map['Clean_Zip'] = df_map[dict_zip_col].astype(str).str.zfill(5)
    
    df = pd.merge(df_sales, df_map, on='Clean_Zip', how='inner')
    
    # Step 1: Data Trimming
    dma_totals = df.groupby(dma_col)[sales_col].sum().sort_values(ascending=False)
    
    if len(dma_totals) > 110: 
        valid_dmas = dma_totals.iloc[10:-100].index.tolist()
        trim_msg = f"Started with {len(dma_totals)} DMAs. Removed the Top 10 and Bottom 100. **{len(valid_dmas)} DMAs** remain for pairing."
        trim_success = True
    else:
        valid_dmas = dma_totals.index.tolist()
        trim_msg = f"Only found {len(dma_totals)} DMAs. Not enough to safely trim without losing everything."
        trim_success = False
        
    df_filtered = df[df[dma_col].isin(valid_dmas)]
    
    # Create Daily Pivot
    daily_pivot = df_filtered.pivot_table(index=date_col, columns=dma_col, values=sales_col, aggfunc='sum').fillna(0)
    
    def find_pairs(df_pivot, min_corr):
        corr_matrix = df_pivot.corr()
        corr_matrix.index.name = None
        corr_matrix.columns.name = None
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = upper_tri.stack().reset_index()
        corr_pairs.columns = ['DMA_1', 'DMA_2', 'Correlation']
        corr_pairs = corr_pairs[corr_pairs['Correlation'] >= min_corr].sort_values('Correlation', ascending=False)
        
        pairs = []
        paired = set() 
        
        for _, row in corr_pairs.iterrows():
            d1, d2, corr = row['DMA_1'], row['DMA_2'], row['Correlation']
            if d1 not in paired and d2 not in paired:
                roles = random.sample([d1, d2], 2) # Random coin flip!
                pairs.append({
                    'Treatment_DMA': roles[0],
                    'Control_DMA': roles[1],
                    'Correlation': round(corr, 4)
                })
                paired.update([d1, d2])
        return pairs, paired

    # Pass 1: Daily Matches
    daily_pairs, daily_paired_dmas = find_pairs(daily_pivot, min_corr)
    for p in daily_pairs: p['Matched_On'] = 'Daily'
    
    # Pass 2: Weekly Fallback
    leftover_dmas = [d for d in valid_dmas if d not in daily_paired_dmas]
    weekly_pairs = []
    
    if len(leftover_dmas) > 1:
        weekly_pivot = daily_pivot[leftover_dmas].resample('W-MON').sum()
        weekly_pairs, weekly_paired_dmas = find_pairs(weekly_pivot, min_corr)
        for p in weekly_pairs: p['Matched_On'] = 'Weekly'

    all_pairs = daily_pairs + weekly_pairs
    results_df = pd.DataFrame(all_pairs)
    
    if not results_df.empty:
        results_df.index = results_df.index + 1
        results_df.index.name = 'Pair_ID'
        results_df = results_df.reset_index()
        
    return results_df, daily_pivot, leftover_dmas, trim_msg, trim_success

# --- MAIN LOGIC ---
if sales_file and zip_dma_file:
    with st.spinner("Processing data..."):
        # Load Data
        df_sales_raw, df_map_raw = load_data(sales_file, zip_dma_file)
        
        # Run Cached Processing
        results_df, daily_pivot, leftover_dmas, trim_msg, trim_success = process_data(
            df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr
        )
        
    # Step 1: Data Trimming
    st.header("Step 1: Outlier Trimming")
    if trim_success:
        st.success(trim_msg)
    else:
        st.warning(trim_msg)
        
    # Step 2: Correlation Results
    st.header("Step 2: Correlation & Pairing Results")
    if not results_df.empty:
        daily_count = len(results_df[results_df['Matched_On'] == 'Daily'])
        weekly_count = len(results_df[results_df['Matched_On'] == 'Weekly'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pairs Found", len(results_df))
        col2.metric("Matched on Daily", daily_count)
        col3.metric("Matched on Weekly Fallback", weekly_count)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Step 3: Visual Validation
        st.header("Step 3: Visual Validation")
        st.markdown("Select a pair below to visually verify their historical trends move together.")
        
        pair_options = results_df.apply(lambda x: f"Pair {x['Pair_ID']}: {x['Treatment_DMA']} vs {x['Control_DMA']} ({x['Matched_On']}, r={x['Correlation']})", axis=1).tolist()
        selected_pair_str = st.selectbox("Select pair to plot:", pair_options)
        
        selected_idx = pair_options.index(selected_pair_str)
        t_dma = results_df.iloc[selected_idx]['Treatment_DMA']
        c_dma = results_df.iloc[selected_idx]['Control_DMA']
        matched_on = results_df.iloc[selected_idx]['Matched_On']
        
        plot_data = daily_pivot if matched_on == 'Daily' else daily_pivot[leftover_dmas].resample('W-MON').sum()
        chart_data = plot_data[[t_dma, c_dma]].reset_index()
        
        fig = px.line(chart_data, x=date_col, y=[t_dma, c_dma], 
                      title=f"Historical Sales ({matched_on}): Treatment vs Control",
                      labels={'value': 'Gross Sales', 'variable': 'Assignment'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Step 4: Export
        st.header("Step 4: Export Test Design")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Paired Markets (CSV)", data=csv, file_name='geo_test_pairs.csv', mime='text/csv')
    else:
        st.error(f"No pairs found with a correlation above {min_corr}. Try lowering the threshold.")
else:
    st.info("👈 Please upload your Shopify Sales and Zip-to-DMA Dictionary in the sidebar to begin.")