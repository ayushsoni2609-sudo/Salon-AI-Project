import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Salon AI Pro", layout="wide")

def load_data():
    c = pd.read_csv('salon_customers_10k.csv')
    h = pd.read_csv('customer_service_history_10k.csv')
    m = pd.merge(c, h, on='customer_id')
    return c, m

try:
    df, merged_df = load_data()
    s_col = 'service_taken' 
    st.title("✂️ AI Salon Automation & Analytics")
    
    # --- CHARTS SECTION (Recommendation se pehle dikhega) ---
    st.header("📊 Business Insights")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Top 5 Services")
        service_counts = merged_df[s_col].value_counts().head(5)
        st.bar_chart(service_counts)
        
    with col_chart2:
        st.subheader("Customer Categories")
        cat_counts = df['category'].value_counts()
        st.bar_chart(cat_counts) # Isse categories ka graph dikhega

    st.markdown("---")

    matrix = merged_df.groupby(['customer_id', s_col]).size().unstack(fill_value=0)
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    st.title("✂️ AI Salon Automation Dashboard")
    st.success("✅ System Online: Database Loaded")
    st.sidebar.title("🌟 Project Info")
    st.sidebar.info("Developed by: Ayush")
    st.sidebar.markdown("---") # Ye ek line draw karega
    # --- Sidebar Styling ---

    st.sidebar.header("Search Customer")
    user_num = st.sidebar.number_input("Enter ID (1-10000)", min_value=1, max_value=10000, value=1)
    target_id = f"C{str(user_num).zfill(5)}"
    
    if st.sidebar.button("Get AI Suggestion"):
        customer = df[df['customer_id'] == target_id]
        if not customer.empty:
            name = customer.iloc[0]['name']
            st.subheader(f"👋 Welcome, {name}")
            
            if target_id in matrix.index:
                scores = sim_df[target_id].sort_values(ascending=False)
                similar_user = scores.index[1]
                suggestions = matrix.loc[similar_user][matrix.loc[similar_user] > 0].index.tolist()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("💡 **AI Recommended Service**")
                    st.write(f"Based on trends: **{suggestions[0]}**")
                with c2:
                    st.warning("📩 **Marketing Message**")
                    st.code(f"Hi {name}, our AI suggests a {suggestions[0]} for you. Book today for 20% off!")
            else:
                st.info("New customer! Suggesting: Trending Haircut")
        else:
            st.error(f"Customer {target_id} not found.")
except Exception as e:
    st.error(f"Error: {e}")


    