import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def aggregate_shap_values(mean_shap, transformed_names, original_names, mapping):
    agg_dict = {orig: 0.0 for orig in original_names}
    for shap_val, trans_name in zip(mean_shap, transformed_names):
        orig = mapping.get(trans_name)
        if orig is not None:
            agg_dict[orig] += shap_val
        else:
            if trans_name in original_names:
                agg_dict[trans_name] += shap_val
    aggregated = [agg_dict[orig] for orig in original_names]
    return aggregated

def plot_similar_houses(similar_df):
    fig = px.scatter(
        similar_df,
        x="area",
        y="price",
        color="location",
        hover_data=["bedrooms"],
        title="Similar Houses Comparison"
    )
    return fig

def plot_recommendation_gauge(recommendation_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=recommendation_score,
        title={'text': "House Recommendation Score"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    return fig

def plot_boxplot(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"Boxplot of {y_col} by {x_col}")
    return fig

def plot_shap_waterfall(base_value, contributions, feature_names, predicted_value):
    x_labels = ["Base Value"] + feature_names + ["Predicted Price"]
    measures = ["absolute"] + ["relative"] * len(contributions) + ["total"]
    y_values = [base_value] + contributions + [predicted_value]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=x_labels,
        y=y_values,
        measure=measures,
        text=[f"{y:.2f}" for y in y_values],
        textposition="outside",
        connector={"line": {"color": "#2e7bcf"}},
        increasing={"marker": {"color": "#1f77b4"}},
        decreasing={"marker": {"color": "#d62728"}}
    ))
    fig.update_layout(
        title="SHAP Value Waterfall Breakdown",
        plot_bgcolor="white",
        showlegend=False,
        waterfallgap=0.3,
        margin=dict(t=60)
    )
    return fig

def create_shap_table(feature_names, feature_values, shap_values):
    # If lengths don't match, trim shap_values.
    if len(shap_values) != len(feature_names):
        print("Warning: shap_values length doesn't match feature_names. Trimming shap_values.")
        shap_values = shap_values[:len(feature_names)]
    
    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": feature_values,
        "Impact": shap_values
    })
    return df.sort_values("Impact", ascending=False)

def plot_shap_scatter(shap_values, feature_names):
    # If shap_values is an object with a .values attribute, use it; otherwise assume it's an array.
    if hasattr(shap_values, 'values'):
        data = shap_values.values
    else:
        data = shap_values

    df_shap = pd.DataFrame(data, columns=feature_names)
    df_melt = df_shap.melt(var_name="Feature", value_name="SHAP Value")
    fig = px.strip(
        df_melt,
        x="Feature",
        y="SHAP Value",
        color="Feature",
        title="Individual SHAP Value Distribution per Feature",
        template="plotly_white"
    )
    fig.update_traces(jitter=0.3)
    return fig

def plot_histogram(df, column, nbins=20, title=None, color_sequence=["#2e7bcf"]):
    title = title or f"Distribution of {column}"
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        color_discrete_sequence=color_sequence
    )
    return fig

def plot_box(df, column, title=None, color_sequence=["#2e7bcf"]):
    title = title or f"Boxplot of {column}"
    fig = px.box(
        df,
        y=column,
        title=title,
        color_discrete_sequence=color_sequence
    )
    return fig