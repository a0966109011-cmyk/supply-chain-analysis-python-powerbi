"""
DataCo Supply Chain Dataset - 資料清理腳本
================================================
用途：清理原始供應鏈資料，輸出乾淨版本供後續 EDA / Power BI 使用
資料來源：DataCo Supply Chain Dataset (Kaggle)
"""

import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------
# 路徑設定：用腳本自身的位置當基準，不依賴「執行時的工作目錄」
# -----------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# -----------------------------------------------------------------
# 1. 讀取資料
#    注意：這份資料是 ISO-8859-1 編碼，不是 UTF-8，
#    用預設編碼讀取會直接 UnicodeDecodeError
# -----------------------------------------------------------------
INPUT_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"
OUTPUT_PATH = BASE_DIR / "DataCo_cleaned.csv"

df = pd.read_csv(INPUT_PATH, encoding="ISO-8859-1")
print(f"原始資料：{df.shape[0]:,} 筆，{df.shape[1]} 個欄位")

# -----------------------------------------------------------------
# 2. 刪除不需要的欄位
#    - Product Description：整欄都是空值
#    - Order Zipcode：缺失率高達 86%，分析價值低
#    - 個資相關欄位：Email / Password / 姓名 / 街道地址
#      （這份雖是模擬資料，但公開到 GitHub 上仍建議移除，
#       一來不影響分析，二來作品集公開呈現更專業）
# -----------------------------------------------------------------
columns_to_drop = [
    "Product Description",
    "Order Zipcode",
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Product Image",  # 圖片網址，分析用不到
]
df = df.drop(columns=columns_to_drop)

# -----------------------------------------------------------------
# 3. 轉換日期欄位
#    原始格式：M/D/YYYY HH:MM（字串），轉成 datetime 才能做時間運算
# -----------------------------------------------------------------
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")
df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"])

# 額外衍生欄位：年、月、星期幾（EDA 階段做季節性分析會用到）
df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_weekday"] = df["order_date"].dt.day_name()

# -----------------------------------------------------------------
# 4. 處理少量缺失值
#    Customer Zipcode 只缺 3 筆，直接捨棄這幾筆即可，
#    不需要為了 3 筆資料做填補
# -----------------------------------------------------------------
before = len(df)
df = df.dropna(subset=["Customer Zipcode"])
print(f"因 Customer Zipcode 缺失捨棄了 {before - len(df)} 筆")

# -----------------------------------------------------------------
# 5. 欄位名稱標準化（方便後續 Power BI / Python 取用，避免空格與括號）
# -----------------------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# -----------------------------------------------------------------
# 6. 基本邏輯檢查（異常值）
#    Days for shipping (real) 理論上不該是負數
# -----------------------------------------------------------------
neg_shipping = (df["Days_for_shipping_real"] < 0).sum()
if neg_shipping > 0:
    print(f"發現 {neg_shipping} 筆運送天數為負數，已篩除")
    df = df[df["Days_for_shipping_real"] >= 0]

# -----------------------------------------------------------------
# 7. 輸出清理後的資料
# -----------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"清理完成：{df.shape[0]:,} 筆，{df.shape[1]} 個欄位")
print(f"已輸出至 {OUTPUT_PATH}")