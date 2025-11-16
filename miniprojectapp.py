import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Telco Churn Analysis", page_icon="📊", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('C:/Users/User/Documents/CSDS Third Year/Applied Data Science/Mini Project/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory.")
        return None

# =========================
# PREPROCESSING FOR TRAINING
# =========================
def preprocess_data(df):
    df = df.copy()
    
    # Keep original categorical columns for filters and display
    df['gender_raw'] = df['gender']
    df['InternetService_raw'] = df['InternetService']
    df['Contract_raw'] = df['Contract']
    df['PaymentMethod_raw'] = df['PaymentMethod']
    
    # Convert TotalCharges to numeric (handle spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Encode categorical columns
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    
    # Encode all internet service related columns
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

# =========================
# CHURN PREDICTION MODEL
# =========================
@st.cache_resource
def train_churn_model(df, model_type='Logistic Regression'):
    df = df.copy()
    
    # Drop raw categorical columns and customerID for training
    cols_to_drop = ['gender_raw', 'InternetService_raw', 'Contract_raw', 'PaymentMethod_raw', 'customerID']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_names = X.columns.tolist()
    if model_type == 'Random Forest':
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return model, scaler, accuracy, conf_matrix, feature_importance, X.columns.tolist()

# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_churn_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Churn count
    churn_counts = df['Churn'].value_counts()
    ax[0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax[0].set_title('Churn Distribution')
    
    # Churn rate by gender
    if 'gender_raw' in df.columns:
        gender_churn = df.groupby('gender_raw')['Churn'].mean()
        gender_churn.plot(kind='bar', ax=ax[1], color=['#3498db', '#e67e22'])
        ax[1].set_title('Churn Rate by Gender')
        ax[1].set_ylabel('Churn Rate')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def plot_churn_by_feature(df, feature, title):
    if feature not in df.columns:
        st.warning(f"Feature '{feature}' not found in the dataset.")
        return None
    
    churn_data = df.groupby(feature)['Churn'].mean().sort_values(ascending=False)
    
    if churn_data.empty:
        st.warning(f"No data available for {feature}.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_data.plot(kind='bar', ax=ax, color='#e74c3c')
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel("Churn Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.axhline(y=df['Churn'].mean(), color='blue', linestyle='--', label='Average Churn Rate')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_numeric_distribution(df, feature):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution by churn
    for churn_val in [0, 1]:
        subset = df[df['Churn'] == churn_val][feature]
        ax[0].hist(subset, bins=30, alpha=0.6, label=f'Churn={churn_val}')
    ax[0].set_title(f'{feature} Distribution by Churn')
    ax[0].set_xlabel(feature)
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    
    # Box plot
    df.boxplot(column=feature, by='Churn', ax=ax[1])
    ax[1].set_title(f'{feature} by Churn Status')
    ax[1].set_xlabel('Churn')
    ax[1].set_ylabel(feature)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

def plot_feature_importance(feature_importance, top_n=10):
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(top_n)
    ax.barh(top_features['feature'], top_features['importance'], color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features Affecting Churn')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# =========================
# MAIN APP
# =========================
def app():
    st.title("📊 Telco Customer Churn Analysis and Prediction")
    st.markdown("### Analyze customer churn patterns and predict future churn")
    
    # Load data
    df_raw = load_data()
    if df_raw is None:
        return
    
    df = preprocess_data(df_raw)
    
    # Sidebar
    st.sidebar.title("🔍 Navigation & Filters")
    page = st.sidebar.radio("Select Page", ["📈 Dashboard", "🔬 Exploratory Analysis", "🤖 Churn Prediction", "📊 Model Performance"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Data Filters")
    
    # Filters
    gender_filter = st.sidebar.selectbox("Gender", options=["All", "Female", "Male"])
    internet_filter = st.sidebar.selectbox("Internet Service", options=["All", "DSL", "Fiber optic", "No"])
    contract_filter = st.sidebar.selectbox("Contract Type", options=["All", "Month-to-month", "One year", "Two year"])
    senior_filter = st.sidebar.selectbox("Senior Citizen", options=["All", "Yes", "No"])
    
    # Apply filters
    df_filtered = df.copy()
    if gender_filter != "All":
        df_filtered = df_filtered[df_filtered['gender_raw'] == gender_filter]
    if internet_filter != "All":
        df_filtered = df_filtered[df_filtered['InternetService_raw'] == internet_filter]
    if contract_filter != "All":
        df_filtered = df_filtered[df_filtered['Contract_raw'] == contract_filter]
    if senior_filter != "All":
        senior_val = 1 if senior_filter == "Yes" else 0
        df_filtered = df_filtered[df_filtered['SeniorCitizen'] == senior_val]
    
    # Display filtered count
    st.sidebar.info(f"Showing {len(df_filtered)} of {len(df)} customers")
    
    # =========================
    # PAGE 1: DASHBOARD
    # =========================
    if page == "📈 Dashboard":
        st.header("Dashboard Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            churn_rate = df_filtered['Churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        with col2:
            total_customers = len(df_filtered)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col3:
            avg_tenure = df_filtered['tenure'].mean()
            st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
        
        with col4:
            avg_monthly = df_filtered['MonthlyCharges'].mean()
            st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
        
        st.markdown("---")
        
        # Churn Distribution
        st.subheader("📊 Churn Distribution")
        fig = plot_churn_distribution(df_filtered)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Rate by Contract Type")
            fig = plot_churn_by_feature(df_filtered, 'Contract_raw', 'Churn Rate by Contract Type')
            if fig:
                st.pyplot(fig)
        
        with col2:
            st.subheader("Churn Rate by Internet Service")
            fig = plot_churn_by_feature(df_filtered, 'InternetService_raw', 'Churn Rate by Internet Service')
            if fig:
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Payment method analysis
        st.subheader("Churn Rate by Payment Method")
        fig = plot_churn_by_feature(df_filtered, 'PaymentMethod_raw', 'Churn Rate by Payment Method')
        if fig:
            st.pyplot(fig)
    
    # =========================
    # PAGE 2: EXPLORATORY ANALYSIS
    # =========================
    elif page == "🔬 Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        # Dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"**Total Rows:** {len(df_raw)}")
            st.write(f"**Total Columns:** {len(df_raw.columns)}")
            st.write(f"**Missing Values:** {df_raw.isnull().sum().sum()}")
        
        with col2:
            st.subheader("Churn Summary")
            churn_summary = df_raw['Churn'].value_counts()
            st.write(f"**No Churn:** {churn_summary.get('No', 0)}")
            st.write(f"**Churn:** {churn_summary.get('Yes', 0)}")
            st.write(f"**Churn Rate:** {(churn_summary.get('Yes', 0) / len(df_raw) * 100):.2f}%")
        
        st.markdown("---")
        
        # Numeric feature analysis
        st.subheader("📈 Numeric Features Analysis")
        numeric_feature = st.selectbox("Select Numeric Feature", 
                                       options=['tenure', 'MonthlyCharges', 'TotalCharges'])
        
        fig = plot_numeric_distribution(df_filtered, numeric_feature)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Correlation with churn
        st.subheader("🔗 Feature Correlation with Churn")
        
        # Calculate correlations
        df_corr = df_filtered.copy()
        cols_to_drop = ['gender_raw', 'InternetService_raw', 'Contract_raw', 'PaymentMethod_raw', 'customerID']
        for col in cols_to_drop:
            if col in df_corr.columns:
                df_corr = df_corr.drop(col, axis=1)
        
        correlations = df_corr.corr()['Churn'].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        correlations.drop('Churn').plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_title('Feature Correlation with Churn')
        ax.set_xlabel('Correlation Coefficient')
        plt.tight_layout()
        st.pyplot(fig)
    
    # =========================
    # PAGE 3: CHURN PREDICTION
    # =========================
    elif page == "🤖 Churn Prediction":
        st.header("Churn Prediction Model")
        
        # Model selection
        model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
        
        # Train model
        with st.spinner("Training model..."):
            model, scaler, accuracy, conf_matrix, feature_importance, feature_names = train_churn_model(df, model_type)
        
        st.success(f"✅ Model trained successfully with **{accuracy*100:.2f}%** accuracy!")
        
        st.markdown("---")
        
        # User input for prediction
        st.subheader("🧮 Predict Customer Churn")
        st.write("Enter customer details to predict churn probability:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_gender = st.selectbox("Gender", ["Female", "Male"])
            input_senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            input_partner = st.selectbox("Partner", ["No", "Yes"])
            input_dependents = st.selectbox("Dependents", ["No", "Yes"])
            input_tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        with col2:
            input_phone = st.selectbox("Phone Service", ["No", "Yes"])
            input_multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            input_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            input_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            input_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        
        with col3:
            input_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            input_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            input_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            input_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
            input_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
        col4, col5 = st.columns(2)
        with col4:
            input_paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            input_payment = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col5:
            input_monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            input_total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
        
        # Prepare input data
        if st.button("🔮 Predict Churn", type="primary"):
            # Encode inputs
            input_data = pd.DataFrame([{
                'gender': 1 if input_gender == 'Male' else 0,
                'SeniorCitizen': 1 if input_senior == 'Yes' else 0,
                'Partner': 1 if input_partner == 'Yes' else 0,
                'Dependents': 1 if input_dependents == 'Yes' else 0,
                'tenure': input_tenure,
                'PhoneService': 1 if input_phone == 'Yes' else 0,
                'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2}[input_multiple],
                'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2}[input_internet],
                'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_security],
                'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_backup],
                'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_protection],
                'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_support],
                'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_tv],
                'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2}[input_movies],
                'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[input_contract],
                'PaperlessBilling': 1 if input_paperless == 'Yes' else 0,
                'PaymentMethod': {
                    'Electronic check': 0, 'Mailed check': 1,
                    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
                }[input_payment],
                'MonthlyCharges': input_monthly,
                'TotalCharges': input_total
            }])
            
            # Ensure correct column order
            input_data = input_data[feature_names]
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **Customer is likely to CHURN**")
                else:
                    st.success("✅ **Customer is likely to STAY**")
            
            with col2:
                st.metric("Churn Probability", f"{prediction_proba[1]*100:.2f}%")
                st.metric("Stay Probability", f"{prediction_proba[0]*100:.2f}%")
    
    # =========================
    # PAGE 4: MODEL PERFORMANCE
    # =========================
    elif page == "📊 Model Performance":
        st.header("Model Performance Analysis")
        
        # Model selection
        model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
        
        # Train model
        with st.spinner("Training model..."):
            model, scaler, accuracy, conf_matrix, feature_importance, feature_names = train_churn_model(df, model_type)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Accuracy")
            st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
            
            st.subheader("🔍 Confusion Matrix")
            fig = plot_confusion_matrix(conf_matrix)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Classification Metrics")
            # Calculate additional metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
                'Value': [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", tp, tn, fp, fn]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        top_n = st.slider("Number of top features to display", 5, 20, 10)
        fig = plot_feature_importance(feature_importance, top_n)
        st.pyplot(fig)
        
        # Feature importance table
        with st.expander("📋 View Full Feature Importance Table"):
            st.dataframe(feature_importance, use_container_width=True)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app()