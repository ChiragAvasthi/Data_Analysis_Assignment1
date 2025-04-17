# 💻 Laptop Price Prediction & Classification

This project involves data analysis and machine learning to **predict laptop prices** and classify them into **high or low price categories** using linear and logistic regression models.

---

## 📂 Dataset

The dataset contains specifications of 823 laptops, with features such as:

- **Brand**
- **Processor details (brand, generation)**
- **RAM (size, type)**
- **Storage (SSD, HDD)**
- **Operating System**
- **Graphic card size**
- **Touchscreen & MS Office availability**
- **Price, Ratings, Reviews**

---

## 🧼 Preprocessing Steps

1. Converted RAM, SSD, HDD, and Graphic Card columns from strings like `"8 GB"` to integers.
2. Extracted numeric rating from strings like `"3 stars"` → `3`.
3. Dropped or encoded non-numeric features:
   - One-hot encoded: `brand`, `processor_brand`, `ram_type`, `processor_gnrtn`
   - Binary encoded: `Touchscreen`, `msoffice`
4. Removed unusable columns such as `weight`, `warranty`, `os`, and `processor_name`.

---

## 📊 Visualizations

The project includes:
- Scatter plots of **RAM vs Price**, **SSD vs Price**
- A **correlation heatmap** to see how each feature relates to price

---

## 📈 Linear Regression (Predicting Price)

Used `LinearRegression` from `scikit-learn` to predict the price of a laptop given its features.

**Evaluation Metrics:**
- R² Score
- RMSE (Root Mean Square Error)

---

## 📘 Logistic Regression (Classifying Price Level)

Defined a new column `high_price`:
- `1` = Price above the median
- `0` = Price at or below the median

Trained a `LogisticRegression` model to classify laptops as high or low priced.

**Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## 🛠️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
