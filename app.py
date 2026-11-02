import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="Starbucks Beverage Analysis", page_icon="☕", layout="wide")
st.title("☕ Starbucks Beverage Analysis")
st.write("Explore Starbucks nutrition data, visualizations, and Decision Tree predictions.")

sns.set_theme(style="whitegrid")


@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent

    # First look for the CSV directly beside app.py.
    csv_path = base / "starbucks.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Otherwise, look inside archive.zip beside app.py.
    zip_path = base / "archive.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_names = [
                name for name in archive.namelist()
                if Path(name).name.lower() == "starbucks.csv"
            ]
            if csv_names:
                with archive.open(csv_names[0]) as file:
                    return pd.read_csv(file)

    raise FileNotFoundError(
        "Could not find starbucks.csv or archive.zip. "
        "Upload archive.zip (containing starbucks.csv) beside app.py in GitHub."
    )


try:
    df = load_data()
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
st.success("Dataset loaded successfully.")

st.header("1. Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df))
c2.metric("Unique Beverages", df["Beverage"].nunique() if "Beverage" in df else "N/A")
c3.metric(
    "Beverage Categories",
    df["Beverage_category"].nunique() if "Beverage_category" in df else "N/A"
)

with st.expander("View dataset"):
    st.dataframe(df.head(20), use_container_width=True)
with st.expander("Missing values"):
    st.dataframe(df.isnull().sum().rename("Missing Values").to_frame())
with st.expander("Descriptive statistics"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)

st.header("2. Data Cleaning")
df_clean = df.copy()

features = [
    "Calories", "Total Fat (g)", "Trans Fat (g)", "Saturated Fat (g)",
    "Sodium (mg)", "Total Carbohydrates (g)", "Cholesterol (mg)",
    "Dietary Fibre (g)", "Sugars (g)", "Protein (g)", "Caffeine (mg)"
]
target = "Beverage_category"
required = ["Beverage", "Beverage_category"] + features
missing = [column for column in required if column not in df_clean.columns]
if missing:
    st.error("Required dataset columns are missing: " + ", ".join(missing))
    st.stop()

# Convert model features to numeric values, handling text such as "Varies".
for column in features:
    df_clean[column] = pd.to_numeric(
        df_clean[column].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )
    median = df_clean[column].median()
    df_clean[column] = df_clean[column].fillna(median if pd.notna(median) else 0)

df_clean = df_clean.dropna(subset=["Beverage", target])
st.write("Data cleaning completed. Remaining missing values:")
st.dataframe(df_clean.isnull().sum().rename("Missing Values").to_frame())

st.header("3. Nutrition Summary")
avg_calories = df_clean["Calories"].mean()
avg_sugar = df_clean["Sugars (g)"].mean()
correlation = df_clean["Calories"].corr(df_clean["Sugars (g)"])
m1, m2, m3 = st.columns(3)
m1.metric("Average Calories", f"{avg_calories:.1f} kcal")
m2.metric("Average Sugar", f"{avg_sugar:.1f} g")
m3.metric("Calories-Sugar Correlation", f"{correlation:.2f}")

highest = df_clean.loc[df_clean["Calories"].idxmax()]
lowest = df_clean.loc[df_clean["Calories"].idxmin()]
sugariest = df_clean.loc[df_clean["Sugars (g)"].idxmax()]
st.write(f"**Highest-calorie beverage:** {highest['Beverage']} — {highest['Calories']} kcal")
st.write(f"**Lowest-calorie beverage:** {lowest['Beverage']} — {lowest['Calories']} kcal")
st.write(f"**Highest-sugar beverage:** {sugariest['Beverage']} — {sugariest['Sugars (g)']} g")

st.header("4. Data Visualizations")
numeric_columns = df_clean.select_dtypes(include=np.number).columns.tolist()
selected = st.selectbox("Choose a column for its distribution", numeric_columns)
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df_clean[selected], kde=True, ax=ax)
ax.set_title(f"Distribution of {selected}")
st.pyplot(fig)
plt.close(fig)

nutrition_columns = [
    "Calories", "Total Fat (g)", "Sodium (mg)", "Total Carbohydrates (g)",
    "Cholesterol (mg)", "Sugars (g)", "Protein (g)", "Caffeine (mg)"
]
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_clean[nutrition_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Nutrition Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)
with right:
    top = df_clean.groupby("Beverage")["Calories"].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top 10 Beverages by Average Calories")
    ax.set_xlabel("Average Calories")
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x="Sugars (g)", y="Calories", hue=target, ax=ax)
    ax.set_title("Sugar vs Calories")
    st.pyplot(fig)
    plt.close(fig)
with right:
    category_avg = df_clean.groupby(target)["Calories"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    category_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Calories by Beverage Category")
    ax.set_ylabel("Average Calories")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

st.header("5. Decision Tree Classification")
X = df_clean[features]
y = df_clean[target]
if y.nunique() < 2:
    st.error("At least two beverage categories are required to train the classifier.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.subheader("Classification Report")
report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    model,
    feature_names=features,
    class_names=[str(label) for label in model.classes_],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
ax.set_title("Starbucks Beverage Category Decision Tree")
st.pyplot(fig)
plt.close(fig)

st.subheader("Actual vs Predicted Categories")
sample = X_test.head(10)
comparison = df_clean.loc[sample.index, ["Beverage"]].copy()
if "Beverage_prep" in df_clean.columns:
    comparison["Prep"] = df_clean.loc[sample.index, "Beverage_prep"]
comparison["Actual Category"] = y_test.loc[sample.index]
comparison["Predicted Category"] = model.predict(sample)
st.dataframe(comparison.reset_index(drop=True), use_container_width=True)

st.success("Analysis and model training completed successfully.")
