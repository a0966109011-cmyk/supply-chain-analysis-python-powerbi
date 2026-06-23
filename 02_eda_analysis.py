# %% [markdown]
# # DataCo Supply Chain — 延遲交付根因分析 (Late Delivery Root Cause Analysis)
#
# **分析目標**：找出 54.8% 訂單延遲的真正原因，並量化政策調整後的潛在改善幅度。
#
# **分析架構**：
# 1. 描述性統計（Descriptive）— 延遲現象有多嚴重
# 2. 根因排除驗證（Diagnostic）— 用分群統計與相關係數排除混淆因子（地區/類別/訂單規模）
# 3. SLA Gap Analysis（Diagnostic + Prescriptive）— 找出真正根因並量化政策調整的影響
#
# 註：圖表標題與軸標籤統一使用英文（這份作品集主要對應海外職缺申請場景），
#     程式碼註解與分析推論說明維持中文方便自己複查邏輯。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110

# -----------------------------------------------------------------
# 路徑設定：用腳本自身的位置當基準，不依賴「執行時的工作目錄」
# 不管是用 Sublime 的 Build、終端機 cd 進其他資料夾再跑、還是
# Jupyter 啟動，charts/ 資料夾永遠會建立在這個 .py 檔案所在的
# 同一個專案資料夾裡，不會因為執行方式不同而跑到別的地方
# -----------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # 在 Jupyter Notebook 環境下沒有 __file__，改用目前工作目錄
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_CSV = BASE_DIR / "DataCo_cleaned.csv"
df = pd.read_csv(INPUT_CSV, parse_dates=["order_date", "shipping_date"])
print(f"資料筆數：{len(df):,}　欄位數：{df.shape[1]}")

# %% [markdown]
# ## 1. 描述性統計：延遲現象的整體輪廓

# %%
delivery_status_pct = df["Delivery_Status"].value_counts(normalize=True).mul(100).round(1)
print(delivery_status_pct)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#d64545" if s == "Late delivery" else "#8aa1b1" for s in delivery_status_pct.index]
ax.bar(delivery_status_pct.index, delivery_status_pct.values, color=colors)
ax.set_ylabel("Share of Orders (%)")
ax.set_title("Delivery Status Overview: 54.8% of Orders Arrive Late", fontsize=13, fontweight="bold")
for i, v in enumerate(delivery_status_pct.values):
    ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_delivery_status_overview.png")
plt.show()

# %% [markdown]
# ## 2. 延遲率依 Shipping Mode 拆解：核心訊號出現

# %%
late_by_mode = (df.groupby("Shipping_Mode")["Late_delivery_risk"].mean() * 100).sort_values(ascending=False)
print(late_by_mode.round(1))

fig, ax = plt.subplots(figsize=(7, 4.5))
late_by_mode.plot(kind="barh", ax=ax, color="#d64545")
ax.set_xlabel("Late Delivery Rate (%)")
ax.set_title("Faster Shipping Modes Have Higher Late Rates", fontsize=13, fontweight="bold")
for i, v in enumerate(late_by_mode.values):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_late_rate_by_shipping_mode.png")
plt.show()

# %% [markdown]
# ## 3. 根因排除驗證 (A)：地區是否是混淆因子？
#
# **檢驗邏輯**：若延遲率在不同地區之間差異很大，代表問題可能來自地區物流能力，而非
# Shipping Mode 本身。用「同一 Shipping Mode 在各地區延遲率的標準差」來檢驗一致性——
# 標準差小，代表這個 Shipping Mode 的延遲表現不受地區影響，是穩定、結構性的問題。

# %%
pivot_region = df.pivot_table(
    index="Order_Region", columns="Shipping_Mode", values="Late_delivery_risk", aggfunc="mean"
) * 100
region_std = pivot_region.std().round(2)
print("各 Shipping Mode 在不同地區的延遲率標準差：")
print(region_std)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pivot_region, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "Late Rate (%)"}, ax=ax)
ax.set_title("Late Rate is Consistent Across Regions (Not Region-Driven)", fontsize=13, fontweight="bold")
ax.set_xlabel("Shipping Mode")
ax.set_ylabel("Order Region")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_heatmap_region_vs_shipping_mode.png")
plt.show()

# %% [markdown]
# ## 4. 根因排除驗證 (B)：商品類別、訂單規模是否是混淆因子？

# %%
cat_counts = df["Category_Name"].value_counts()
valid_cats = cat_counts[cat_counts > 1000].index
df_valid_cat = df[df["Category_Name"].isin(valid_cats)]

pivot_cat = df_valid_cat.pivot_table(
    index="Category_Name", columns="Shipping_Mode", values="Late_delivery_risk", aggfunc="mean"
) * 100
cat_std = pivot_cat.std().round(2)
print("各 Shipping Mode 在不同商品類別的延遲率標準差：")
print(cat_std)

corr_cols = ["Late_delivery_risk", "Sales", "Order_Item_Quantity", "Benefit_per_order"]
corr_result = df[corr_cols].corr()["Late_delivery_risk"].drop("Late_delivery_risk").round(3)
print("\n延遲風險與訂單規模相關係數（接近 0 代表無關）：")
print(corr_result)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
region_std.plot(kind="bar", ax=axes[0], color="#5b8fb0")
axes[0].set_title("Std. Dev. Across Regions", fontsize=11)
axes[0].set_ylabel("Late Rate Std. Dev. (pp)")
axes[0].tick_params(axis="x", rotation=20)

cat_std.plot(kind="bar", ax=axes[1], color="#5b8fb0")
axes[1].set_title("Std. Dev. Across Product Categories", fontsize=11)
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle(
    "Confounder Check: Low Variance in Both Dimensions -> Structural Issue, Not Region/Category",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_confounder_check_std.png")
plt.show()

# %% [markdown]
# ## 5. 根因確認：SLA 承諾天數 vs 實際運送天數

# %%
sla_summary = df.groupby("Shipping_Mode").agg(
    Promised_Days=("Days_for_shipment_scheduled", "mean"),
    Actual_Days=("Days_for_shipping_real", "mean"),
).round(2)
sla_summary["Gap"] = (sla_summary["Actual_Days"] - sla_summary["Promised_Days"]).round(2)
print(sla_summary)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(sla_summary))
width = 0.35
ax.bar(x - width / 2, sla_summary["Promised_Days"], width, label="Promised Days (SLA)", color="#8aa1b1")
ax.bar(x + width / 2, sla_summary["Actual_Days"], width, label="Actual Avg. Days", color="#d64545")
ax.set_xticks(x)
ax.set_xticklabels(sla_summary.index, rotation=15)
ax.set_ylabel("Days")
ax.set_title("Root Cause: SLA Promises Are Systematically Below Actual Capability", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_sla_vs_actual.png")
plt.show()

# %% [markdown]
# ## 6. SLA Gap Analysis：政策模擬
#
# 把不合理的承諾天數調整至貼近實際營運能力後，重新計算延遲率。
# - First Class：1 → 2 天
# - Second Class：2 → 4 天
# - Same Day／Standard Class：維持不變（本來就接近準時）

# %%
new_schedule = {"First Class": 2, "Second Class": 4, "Same Day": 0, "Standard Class": 4}
df["new_scheduled_days"] = df["Shipping_Mode"].map(new_schedule)
df["new_late_delivery_risk"] = (df["Days_for_shipping_real"] > df["new_scheduled_days"]).astype(int)

original_rate = df["Late_delivery_risk"].mean() * 100
new_rate = df["new_late_delivery_risk"].mean() * 100
print(f"調整前整體延遲率：{original_rate:.1f}%")
print(f"調整後整體延遲率：{new_rate:.1f}%")
print(f"改善幅度：{original_rate - new_rate:.1f} 個百分點")

before_after = pd.DataFrame({
    "Before": df.groupby("Shipping_Mode")["Late_delivery_risk"].mean() * 100,
    "After": df.groupby("Shipping_Mode")["new_late_delivery_risk"].mean() * 100,
}).round(1)
print(before_after)

fig, ax = plt.subplots(figsize=(8, 5))
before_after.plot(kind="bar", ax=ax, color=["#d64545", "#4caf6e"])
ax.set_ylabel("Late Delivery Rate (%)")
ax.set_title(f"Policy Simulation: Overall Late Rate {original_rate:.1f}% -> {new_rate:.1f}%", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=15)
ax.legend(title="SLA Scenario")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_policy_simulation.png")
plt.show()

# %% [markdown]
# ## 7. 補充：運送天數的變異程度（穩定性風險）
#
# 平均值相同不代表風險相同——同樣是「平均 X 天」，變異程度（穩定 vs 不穩定）
# 本身就是另一層應該揭露的風險訊號。
#
# 注意：First Class 的實際運送天數變異為 0（100% 都是 2 天），這在真實世界物流中
# 並不常見，可能反映此資料集本身有一定程度的模擬/簡化性質——這點誠實揭露，
# 比硬要把它解讀成「真實世界穩定性」更符合分析倫理。

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for mode in df["Shipping_Mode"].unique():
    subset = df[df["Shipping_Mode"] == mode]["Days_for_shipping_real"]
    if subset.nunique() > 1:
        sns.kdeplot(subset, ax=ax, label=mode, fill=True, alpha=0.25)
    else:
        ax.axvline(subset.iloc[0], linestyle="--", label=f"{mode} (constant = {subset.iloc[0]} days)")
ax.set_xlabel("Actual Shipping Days")
ax.set_title("Distribution of Actual Shipping Days by Mode (Stability Risk)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_shipping_days_distribution.png")
plt.show()

# %% [markdown]
# ## 結論摘要
#
# 1. 整體延遲率 54.8%，遠高於一般供應鏈基準
# 2. 延遲率與 Shipping Mode 高度相關，且在 22 個地區、14 個主要商品類別中都一致（標準差 < 3.3pp），
#    訂單規模（Sales/Quantity）與延遲幾乎無相關（|r| < 0.005）— 排除混淆因子
# 3. 根因明確：First Class／Second Class 的 SLA 承諾天數系統性低於實際營運能力
# 4. 政策模擬：調整 SLA 承諾天數後，整體延遲率可由 54.8% 降至 34.1%（改善 20.7 個百分點）
# 5. 額外風險訊號：Second Class 不只延遲率高，運送天數變異也大（2-6 天均勻分布），代表
#    這個選項的「不可預測性」風險高於 First Class
# 6. 限制：資料集部分欄位（如 First Class 的運送天數）變異度為 0，可能反映模擬資料的
#    簡化性質，實際應用時建議以更長期、更分散的真實資料重新驗證

print("\n分析完成，圖表已輸出至 ./charts/ 資料夾")