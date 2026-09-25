# 🍔 Online Food Delivery — Data Analytics Dashboard

An interactive Streamlit dashboard for exploring and analyzing an online food delivery customer dataset — covering data quality checks, sales metrics, group-wise summaries, visual charts, and data-driven business recommendations.

## 📊 Dataset

- **File:** `online food delivery dataset.csv`
- **Records:** 388 customer entries
- **Features:** Age, Gender, Marital Status, Occupation, Monthly Income, Educational Qualifications, Family Size, Customer Type, Latitude, Longitude, Pin Code, Output (ordered online), Feedback

> Add your dataset link here if it's hosted externally (e.g., Kaggle/UCI source URL).

## 📝 Project Description

This project analyzes customer behavior around online food ordering. It loads the raw dataset, performs cleaning and quality checks, engineers derived metrics (e.g., a Sales Score based on gender and age), and presents everything through a multi-tab interactive dashboard built with Streamlit. The goal is to surface patterns in customer demographics and feedback that can inform business decisions for a food delivery service.

### Key Features

- **Dataset Overview** — raw data preview with live filtering (gender, occupation, age range, customer type)
- **Data Quality Report** — null counts, data types, and unique value summaries per column
- **Sales Metrics** — computed Sales Score (Gender × Age) and related KPIs
- **Group Summaries** — aggregated statistics across demographic segments
- **Charts** — visual breakdowns using Matplotlib and Seaborn
- **Business Decisions** — narrative insights and recommendations derived from the analysis

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit** — interactive web dashboard
- **Pandas** / **NumPy** — data loading, cleaning, and processing
- **Matplotlib** / **Seaborn** — data visualization

## ⚙️ Setup & Run Instructions

### 1. Clone/download the project

Place `app.py`, `online food delivery dataset.csv`, and `requirements.txt` in the same folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

If `streamlit` is not recognized as a command (common on Windows), run it as a Python module instead:

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`. If not, open that URL manually.

## 📁 Project Structure

```
Online Food Delivery/
├── app.py                              # Streamlit dashboard application
├── online food delivery dataset.csv    # Source dataset
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 📌 Notes

- The CSV filename must match exactly what's referenced in `app.py` (`CSV_PATH` variable), including spaces.
- Data is cached with `@st.cache_data` for faster reloads on filter changes.
