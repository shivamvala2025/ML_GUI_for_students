import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
# from eli5.sklearn import PermutationImportance
import numpy as np
import io

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO


# Streamlit app layout
st.set_page_config(page_title="RPDS Toolkit", layout="wide")
st.title("Rpds Toolkit")

def load_data():
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        data = pd.read_excel(uploaded_file)
        st.session_state["data"] = data
        st.success("Data Loaded Successfully")



def display_statistics():
    if "data" in st.session_state:
        data = st.session_state["data"]
        threshold = 0.5  # 50% threshold for dropping null-dominant columns

        # Drop columns where more than 50% of values are NULL
        data = data.loc[:, data.isnull().mean() < threshold]

        # Tab Layout
        tab1, tab2 = st.tabs(["📊 Numerical Statistics", "📋 Categorical Statistics"])

        # Numerical Statistics
        with tab1:
            numeric_stats = data.describe(percentiles=[0.25, 0.5, 0.75]).T
            numeric_stats["Mode"] = data.mode().iloc[0]

            # Round values to 2 decimal places
            numeric_stats = numeric_stats.round(2)

            # Convert to HTML with styling
            styled_table = numeric_stats.to_html(classes="styled-table")

            # Custom CSS for bold headers and colors
            st.markdown(
                """
                <style>
                .styled-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 16px;
                    text-align: center;
                }
                .styled-table thead th {
                    background-color: #ff4b4b;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    text-align: center;
                }
                .styled-table tbody td {
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }
                .styled-table tbody tr:hover {
                    background-color: #f1f1f1;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )

            # Display the styled table
            st.markdown(styled_table, unsafe_allow_html=True)

                    # 📌 **Statistical Term Explanations**
            st.markdown(
                """
                
                ### 📌 **Understanding Statistical Metrics in Process Industries**
                <div style='font-size:20px;'>

                - **Count**:  
                  - *Definition*: The number of data points in the dataset (excluding missing values).  
                  - *Significance*: A higher count improves the reliability of process monitoring and ensures that conclusions drawn about the process are statistically robust.
                  
                - **Mean (Average)**:  
                  - *Definition*: The sum of all values divided by the count.  
                  - *Significance*: Represents the central tendency of the process variable. In process industries, the mean is critical for benchmarking normal operating conditions.
                  
                - **Standard Deviation (std)**:  
                  - *Definition*: A measure of the spread or dispersion of the data values.  
                  - *Significance*: Indicates process variability. A low standard deviation means the process is consistent, while a high value can signal instability or quality issues.
                  
                - **Minimum (min)**:  
                  - *Definition*: The smallest observed value in the dataset.  
                  - *Significance*: Useful for detecting lower limits of a process variable, which can be important for safety and operational thresholds.
                  
                - **25% (First Quartile, Q1)**:  
                  - *Definition*: The value below which 25% of the data falls.  
                  - *Significance*: Helps in understanding the lower tail of the data distribution, providing insights into underperforming conditions.
                  
                - **50% (Median, Q2)**:  
                  - *Definition*: The middle value of the dataset when it is ordered.  
                  - *Significance*: Offers a robust measure of central tendency, less sensitive to outliers than the mean, which is vital for processes with occasional anomalies.
                  
                - **75% (Third Quartile, Q3)**:  
                  - *Definition*: The value below which 75% of the data falls.  
                  - *Significance*: Reflects the upper end of the data distribution, useful for understanding peak operating conditions.
                  
                - **Maximum (max)**:  
                  - *Definition*: The highest observed value in the dataset.  
                  - *Significance*: Critical for identifying potential process overloads or peaks that might lead to equipment stress or failure.
                  
                - **Mode**:  
                  - *Definition*: The most frequently occurring value in the dataset.  
                  - *Significance*: Indicates the most common operating condition, which can serve as a benchmark for process optimization and consistency.
                """, 
                unsafe_allow_html=True
            )



        # Categorical Statistics
        with tab2:
            categorical_data = data.select_dtypes(include=['object', 'category'])
            if not categorical_data.empty:
                st.markdown("### 📌 **Categorical Data Counts**")
                for col in categorical_data.columns:
                    st.markdown(f"#### 🔹 **{col}**")
                    st.text(categorical_data[col].value_counts().to_string())
            else:
                st.warning("⚠️ No categorical data found.")
    else:
        st.warning("⚠️ Please load data first.")




def scaler_options():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Filter numerical data

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select Scaler
    scaler_choice = st.selectbox("Select a Scaler:", ["Select", "Standard Scaler", "Min-Max Scaler"])

    # Apply the selected scaler
    if scaler_choice == "Standard Scaler":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        st.session_state["scaled_data"] = scaled_data
        st.success("Data scaled using Standard Scaler.")

    elif scaler_choice == "Min-Max Scaler":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        st.session_state["scaled_data"] = scaled_data
        st.success("Data scaled using Min-Max Scaler.")
        
    elif scaler_choice == "Select":
        st.warning("Please select a scaler.")

def linear_regression_model():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    threshold = 0.5  # 50% threshold for dropping null-dominant columns

    # Drop columns where more than 50% of values are NULL
    data = data.loc[:, data.isnull().mean() < threshold]

    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select features (X) and target variable (y)
    all_features = numeric_data.columns.tolist()
    selected_features = st.selectbox("Select Features (X):", options=all_features, key="feature_select")
    target = st.selectbox("Select Target Variable (y):", options=numeric_data.columns.tolist())

    if len(selected_features) == 0 or target == '':
        st.warning("Please select at least one feature and one target variable.")
        return

    # Split data into training and testing sets
    test_size = st.slider("Test Data Percentage:", min_value=1, max_value=99, value=20)
    X = data[selected_features]
    y = data[target].squeeze()  # ensure y is 1D

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Ensure X_train and X_test are 2D arrays
    if X_train.ndim == 1:
        X_train = X_train.values.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.values.reshape(-1, 1)

    # Scale the feature data before performing linear regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()

    if st.button("Run Model"):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Regression equation on scaled data
        slope = model.coef_[0]
        intercept = model.intercept_
        equation = f"{target} = {slope:.2f} * {selected_features} + ({intercept:.2f})"

        st.subheader("Regression Equation")
        st.latex(f"{equation}")

        # Calculate performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Prepare the predictions vs actual plot
        fig_train_test = go.Figure()
        min_val = min(y_train.min(), y_test.min())
        max_val = max(y_train.max(), y_test.max())

        # Train predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_train, y=y_pred_train, mode="markers", 
            name=f"Train {target} Values", marker=dict(color='green')
        ))
        # Test predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_test, y=y_pred_test, mode="markers", 
            name=f"Test {target} Values", marker=dict(color='blue')
        ))
        # 45-degree line
        fig_train_test.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="y = x", line=dict(color="black", dash="dash")
        ))

        r2_values_text = f"R² (Train): {r2_train:.2f} | R² (Test): {r2_test:.2f}"
        fig_train_test.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text=r2_values_text, showarrow=False,
            font=dict(size=14, color="green"), align="center"
        )

        fig_train_test.update_layout(
            title="Training and Testing Predictions vs Actual",
            font=dict(size=23, color="darkblue", family="Arial", weight="bold"),
            xaxis_title=f"Actual {target}",
            yaxis_title=f"Predicted {target}",
            template="plotly_white",
            height=500,
            width=600,
            autosize=False,
            xaxis=dict(
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),
                title=dict(
                    text=f"Actual {target}",
                    font=dict(size=16, color="black", family="Arial", weight="bold")
                )
            ),
            yaxis=dict(
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),
                title=dict(
                    text=f"Predicted {target}",
                    font=dict(size=16, color="black", family="Arial", weight="bold")
                )
            )
        )

        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_train_test, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)



def visualize_data():
    if "data" in st.session_state:
        data = st.session_state["data"]
        threshold = 0.5  # 50% threshold for dropping null-dominant columns

        # Drop columns where more than 50% of values are NULL
        data = data.loc[:, data.isnull().mean() < threshold]

        # Dropdown to select multiple plot types (added Correlation Matrix)
        plot_types = ["Scatter Plot", "Box Plot", "Histogram", "Pairplot", "Correlation Matrix"]
        selected_plots = st.multiselect("📊 Select Plot Types", plot_types)

        if not selected_plots:
            st.warning("⚠️ Please select at least one plot type.")
            return

        # When only Pairplot and/or Correlation Matrix are selected, no manual variable selection is needed.
        if set(selected_plots).issubset({"Pairplot", "Correlation Matrix"}):
            needs_x_y_selection = False
        else:
            needs_x_y_selection = True

        if needs_x_y_selection:
            # Determine if any of the selected plot types require Y values.
            needs_y = any(plot in ["Scatter Plot"] for plot in selected_plots)

            # Let the user decide how many X variables to choose.
            num_x = st.number_input("How many X variables do you want?", min_value=1, value=1, step=1)
            x_columns = [st.selectbox(f"Select X variable {i+1}", data.columns, key=f"x_{i}") for i in range(num_x)]

            # For Y columns, do the same if needed.
            y_columns = []
            if needs_y:
                num_y = st.number_input("How many Y variables do you want?", min_value=1, value=1, step=1)
                y_columns = [st.selectbox(f"Select Y variable {i+1}", data.columns, key=f"y_{i}") for i in range(num_y)]
        else:
            # Use all numerical columns automatically.
            numerical_cols = data.select_dtypes(include=["number"]).columns.tolist()
            if "Pairplot" in selected_plots and len(numerical_cols) < 2:
                st.warning("⚠️ Not enough numerical columns for a Pairplot.")
                return

        # Initialize or retrieve the list to store plot images
        if "plot_images" not in st.session_state:
            st.session_state.plot_images = []

        # Provide two buttons: one to generate plots and one to clear existing plots.
        col1, col2 = st.columns(2)
        with col1:
            generate_plot = st.button("🚀 Generate Plots")
        with col2:
            clear_plots = st.button("🗑️ Clear All Plots")

        # Clear button logic
        if clear_plots:
            st.session_state.plot_images = []
            st.success("Cleared all plots!")
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
            else:
                st.info("Please refresh the page manually to see changes.")

        if generate_plot:
            # Basic input checks for manual selection
            if needs_x_y_selection:
                if not x_columns:
                    st.warning("⚠️ Please select at least one X column.")
                    return
                if needs_y and not y_columns:
                    st.warning("⚠️ Please select at least one Y column.")
                    return

            # Generate plots sequentially
            for selected_plot in selected_plots:
                if selected_plot == "Scatter Plot":
                    for x, y in zip(x_columns, y_columns):
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        sns.scatterplot(data=data, x=x, y=y, ax=ax)
                        ax.set_title(f"{selected_plot}: {x} vs {y}", fontsize=14)
                        plt.tight_layout()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        st.session_state.plot_images.append(buf.getvalue())
                        plt.close(fig)
                elif selected_plot in ["Box Plot", "Histogram"]:
                    for x in x_columns:
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        if selected_plot == "Box Plot":
                            sns.boxplot(data=data, x=x, ax=ax)
                            # Compute descriptive statistics
                            q1 = data[x].quantile(0.25)
                            median = data[x].quantile(0.5)
                            q3 = data[x].quantile(0.75)
                            minimum = data[x].min()
                            maximum = data[x].max()
                            IQR = q3 - q1
                            # Add vertical lines for median, Q1, Q3, min and max
                            ax.axvline(median, color='red', linestyle='--', label='Median')
                            ax.axvline(q1, color='blue', linestyle='--', label='25%')
                            ax.axvline(q3, color='blue', linestyle='--', label='75%')
                            ax.axvline(minimum, color='green', linestyle='--', label='Min')
                            ax.axvline(maximum, color='green', linestyle='--', label='Max')
                            # Annotate IQR
                            ax.annotate(f'IQR: {IQR:.2f}', xy=(median, 0.5), xycoords=('data', 'axes fraction'),
                                        fontsize=10, color='purple')
                        else:
                            sns.histplot(data=data[x], kde=True, bins=20, ax=ax)
                            # Compute mean and median
                            mean_val = data[x].mean()
                            median_val = data[x].median()
                            # Add vertical lines for mean and median
                            ax.axvline(mean_val, color='red', linestyle='--', label='Mean')
                            ax.axvline(median_val, color='green', linestyle='--', label='Median')
                        ax.set_title(f"{selected_plot}: {x}", fontsize=14)
                        ax.legend()
                        plt.tight_layout()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        st.session_state.plot_images.append(buf.getvalue())
                        plt.close(fig)
                elif selected_plot == "Pairplot":
                    pair_fig = sns.pairplot(data[numerical_cols])
                    buf = io.BytesIO()
                    pair_fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.session_state.plot_images.append(buf.getvalue())
                    plt.close('all')
                elif selected_plot == "Correlation Matrix":
                    # Compute the correlation matrix for all numerical columns
                    corr = data.select_dtypes(include=["number"]).corr()
                    fig, ax = plt.subplots(figsize=(7, 6))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Matrix", fontsize=14)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.session_state.plot_images.append(buf.getvalue())
                    plt.close(fig)

            st.success("Plots generated!")

        # Display all accumulated plots
        if "plot_images" in st.session_state and st.session_state.plot_images:
            st.markdown("### Generated Plots")
            for img in st.session_state.plot_images:
                st.image(img, use_container_width=True, width=400)

        # Include explanations for key plot types
        st.markdown(
            """
            ### Significance of Selected Plot Types in Process Industries
            
            **Scatter Plot:**  
            - *Purpose*: Visualizes the relationship between two continuous variables.  
            - *Significance*: In process industries, scatter plots help identify correlations between process parameters (e.g., temperature vs. pressure), which are crucial for process optimization and anomaly detection.
            
            **Box Plot:**  
            - *Purpose*: Summarizes the distribution of a dataset by showing the median, quartiles, and potential outliers.  
            - *Significance*: Box plots are used in process industries for quality control, enabling quick identification of variability, skewness, and outliers that may indicate operational issues. The additional lines for median, 25% and 75% (Q1 & Q3), min, max, and IQR provide deeper insights into process stability.
            
            **Histogram:**  
            - *Purpose*: Displays the frequency distribution of a single variable.  
            - *Significance*: Histograms are essential in process industries for understanding the distribution and variability of key parameters (e.g., production yield, energy consumption), aiding in monitoring process consistency. The added mean and median lines help quickly assess the central tendency and spread of the data.
            
            **Pairplot:**  
            - *Purpose*: Provides a grid of scatter plots for each pair of variables and histograms along the diagonal, offering a comprehensive view of variable relationships.  
            - *Significance*: Pairplots help identify multivariate correlations and potential multicollinearity among process variables, which is critical for understanding complex process dynamics.
            
            **Correlation Matrix:**  
            - *Purpose*: Displays pairwise correlation coefficients between numerical variables using a heatmap.  
            - *Significance*: A correlation matrix is invaluable in process industries for quickly identifying strong interdependencies between process parameters, aiding in troubleshooting, process optimization, and control strategy development.
            """, 
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please load data first.")


def ai_model():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]

    threshold = 0.5  # 50% threshold for dropping null-dominant columns

    # Drop columns where more than 50% of values are NULL
    data = data.loc[:, data.isnull().mean() < threshold]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Filter numerical data

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select features (X) and target variable (y)
    all_features = numeric_data.columns.tolist()
    selected_features = st.multiselect("Select Features (X):", options=all_features, default=all_features)
    # selected_features = st.selectbox("Select Target Variable (X):", options=numeric_data.columns.tolist())

    target = st.selectbox("Select Target Variable (y):", options=numeric_data.columns.tolist())

    if len(selected_features) == 0 or not target:
        st.warning("Please select at least one feature and one target variable.")
        return

    # Split data into training and testing sets
    test_size = st.slider("Test Data Percentage:", min_value=1, max_value=99, value=20)
    train_size = 100 - test_size

    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
    if X_train.ndim == 1:
        X_train = X_train.values.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.values.reshape(-1, 1)
    # Model selection
    model_choice = st.selectbox("Choose an AI Model:", ["Select", "LightGBM", "XGBoost", "Extra Trees"])

    if model_choice == "LightGBM":
        # LightGBM Parameters
        # learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=-1, max_value=50, value=-1, step=1)
        learning_rate = 0.01
        n_estimators = 50
        max_depth = 3

        model = LGBMRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

    elif model_choice == "XGBoost":
        # XGBoost Parameters
        # learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=6, step=1)

        
        learning_rate = 0.01
        n_estimators = 50
        max_depth = 3
        model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif model_choice == "Extra Trees":
        # Extra Trees Parameters
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=None, step=1)
        n_estimators = 50
        max_depth = 3

        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


    else:
        st.warning("Please select a valid model.")
        return

    # Run the model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        min_val = min(y_train.min(), y_test.min())
        max_val = max(y_train.max(), y_test.max())

        # Visualization of metrics
        st.subheader("Model Performance")

        fig_train_test = go.Figure()

        # Train predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_train.values , y=y_pred_train, mode="markers", name=f"Train {target} Values", marker=dict(color='green')
        ))
        


        # Test predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_test.values, y=y_pred_test, mode="markers", name=f"Test {target} Values ", marker=dict(color='blue')
        ))

        fig_train_test.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], 
            mode="lines", name="y = x", 
            line=dict(color="black", dash="dash")
        ))


        # Combine R² values into one annotation
        r2_values_text = f"R² (Train): {r2_train:.2f} | R² (Test): {r2_test:.2f}"
        fig_train_test.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text=r2_values_text, showarrow=False, font=dict(size=14, color="green"),
            align="center"
        )
        

        fig_train_test.update_layout(
            title="Training and Testing Predictions vs Actual",
            font=dict(size=23, color="darkblue", family="Arial", weight="bold"),
            xaxis_title=f"Actual {target}",
            yaxis_title=f"Predicted  {target}",
            template="plotly_white",
            height=500,
            width = 600,
            autosize=False,
            # xaxis=dict(scaleanchor="y")  # Enforce 1:1 aspect ratio
            xaxis=dict(
                # scaleanchor="y",  # Keep square aspect ratio
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),  # X-axis ticks
                title=dict(
                text=f"Actual {target}",
                font=dict(size=16, color="black", family="Arial", weight="bold"),  # X-axis title customization
                )
            ),
            yaxis=dict(
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),  # Y-axis ticks
                title=dict(
                text=f"Predicted {target}",
                font=dict(size=16, color="black", family="Arial", weight="bold"),  # X-axis title customization
                )
                
            )        

        )

        st.markdown(
            "<div style='display: flex; justify-content: center;'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_train_test, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)


def multiple_linear_regression():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    threshold = 0.5  # 50% threshold for dropping columns with too many nulls

    # Drop columns with >50% null values
    data = data.loc[:, data.isnull().mean() < threshold]

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Allow selection of multiple features (X) and one target variable (y)
    all_features = numeric_data.columns.tolist()
    selected_features = st.multiselect("Select Features (X):", options=all_features, default=all_features)
    target = st.selectbox("Select Target Variable (y):", options=all_features)

    if len(selected_features) == 0 or not target:
        st.warning("Please select at least one feature and one target variable.")
        return

    # Ensure the target variable is not included as a feature.
    if target in selected_features:
        selected_features.remove(target)

    test_size = st.slider("Test Data Percentage:", min_value=1, max_value=99, value=20)
    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Scale the feature data (X) before training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())


    if st.button("Train MLR Model"):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Construct the regression equation (on scaled features)
        equation = f"{target} = {model.intercept_:.2f}"
        for coef, feat in zip(model.coef_, selected_features):
            equation += f" + ({coef:.2f} * {feat})"

        st.subheader("Multiple Linear Regression Equation (on scaled features)")
        st.latex(equation)

        # Compute performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # st.markdown(f"**Train R²:** {r2_train:.2f} | **Test R²:** {r2_test:.2f}")
        # st.markdown(f"**Train MSE:** {mse_train:.2f} | **Test MSE:** {mse_test:.2f}")

        fig_train_test = go.Figure()

        # Train predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_train.values , y=y_pred_train, mode="markers", name=f"Train {target} Values", marker=dict(color='green')
        ))
        


        # Test predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_test.values, y=y_pred_test, mode="markers", name=f"Test {target} Values ", marker=dict(color='blue')
        ))

        fig_train_test.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], 
            mode="lines", name="y = x", 
            line=dict(color="black", dash="dash")
        ))


        # Combine R² values into one annotation
        r2_values_text = f"R² (Train): {r2_train:.2f} | R² (Test): {r2_test:.2f}"
        fig_train_test.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text=r2_values_text, showarrow=False, font=dict(size=14, color="green"),
            align="center"
        )

        

        fig_train_test.update_layout(
            title="Training and Testing Predictions vs Actual",
            font=dict(size=23, color="darkblue", family="Arial", weight="bold"),
            xaxis_title=f"Actual {target}",
            yaxis_title=f"Predicted  {target}",
            template="plotly_white",
            height=500,
            width = 600,
            autosize=False,
            # xaxis=dict(scaleanchor="y")  # Enforce 1:1 aspect ratio
            xaxis=dict(
                # scaleanchor="y",  # Keep square aspect ratio
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),  # X-axis ticks
                title=dict(
                text=f"Actual {target}",
                font=dict(size=16, color="black", family="Arial", weight="bold"),  # X-axis title customization
                )
            ),
            yaxis=dict(
                tickfont=dict(size=14, color="black", family="Arial", weight="bold"),  # Y-axis ticks
                title=dict(
                text=f"Predicted {target}",
                font=dict(size=16, color="black", family="Arial", weight="bold"),  # X-axis title customization
                )
                
            )        

        )

        st.markdown(
            "<div style='display: flex; justify-content: center;'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_train_test, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)



def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = ["Home", "Show Basic Statistics","Data Visualization","Scaler", "Linear Regression", "Multiple Linear Regression","AI Model"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.header("Home")
        load_data()

    elif choice == "Show Basic Statistics":
        st.header("Basic Statistics")
        display_statistics()

    elif choice == "Data Visualization":
        visualize_data()

    elif choice == "Scaler":
        st.header("Data Scaling")
        scaler_options()

    elif choice == "Linear Regression":
        st.header("Linear Regression Model")
        linear_regression_model()

    elif choice == "Multiple Linear Regression":
        st.header("Multiple Linear Regression Model")
        multiple_linear_regression()


    elif choice == "AI Model":
        st.header("AI Models")
        ai_model()

if __name__ == "__main__":
    main()
