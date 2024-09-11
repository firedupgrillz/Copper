# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# %%
excel_file = 'V8 data.xlsx'
xls = pd.ExcelFile(excel_file)
sheets_dict = {}
for sheet_name in xls.sheet_names:
    if sheet_name != "_master":  # Skip the "_master" sheet
        sheets_dict[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)

# %%
def parse_weird(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%y')
    

dfs = []
for table, data in sheets_dict.items():
    if table in ["SPX 500", "china", "fed"]: 
        data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")
    elif table in ["Crude", "gold", "SFE"]:
        data["date"] = pd.to_datetime(data["date"], format="%d-%b-%y")
    elif table in ["USD-CNY fx", "GSCI"]: 
        data["date"] = data["date"].apply(parse_weird)
    else: 
        data["date"] = pd.to_datetime(data["date"], format="%d-%b-%Y")

    # rename other columns with prefix
    data = data.rename(columns={c: f"{table}_{c}" for c in data.columns if c != "date"})

    dfs.append(data)


df = dfs[0]
for d in dfs[1:]:
    df = pd.merge(df, d, on="date", how="outer")

df["CHINADAYE_close"] = df["CHINADAYE_close"]*1000

df = df.sort_values("date")
df


# %%
cu_ex_1 = pd.read_csv("table_1.csv")
cu_ex_2 = pd.read_csv("table 5.csv")
# cu_ex_3 = pd.read_csv("table_3.csv")
mine_price = pd.read_csv("table_4.csv")


def rename_cme(col: str) -> str:
    if col == "CME Copper Exchange Date":
        return "date"

    variable_name = col[len("CME Copper ") :].lower().strip()
    variable_name = variable_name.replace(" ", "_")
    variable_name = variable_name.replace("%", "perc_")
    return f"cme_{variable_name}"


def rename_shfe(col: str) -> str:
    if col == "SHFE Copper - Exchange Date":
        return "date"

    variable_name = col[len("SHFE Copper ") :].lower().strip()
    variable_name = variable_name.replace(" ", "_")
    variable_name = variable_name.replace("%", "perc_")
    return f"shfe_{variable_name}"



def rename_mining(col: str) -> str:
    if col == "NYSE / NASDAQ Mining Company Date":
        return "date"

    split_col = col.split(" - ")
    stock = split_col[0].lower().strip()
    variable_name = split_col[1].lower().strip()
    variable_name = variable_name.replace(" ", "_")
    variable_name = variable_name.replace("%", "perc_")
    return f"{stock}_{variable_name}"


# Rename columns
cu_ex_1.rename(columns=rename_cme, inplace=True)
cu_ex_2.rename(columns=rename_shfe, inplace=True)
# cu_ex_3.columns = ["date", "lme_value"]
mine_price.rename(columns=rename_mining, inplace=True)

# Convert date columns to datetime
cu_ex_1["date"] = pd.to_datetime(cu_ex_1["date"], format="%d/%m/%y")
cu_ex_2["date"] = pd.to_datetime(cu_ex_2["date"], format="%d-%b-%y")
# cu_ex_3["date"] = pd.to_datetime(cu_ex_3["date"], format="%d/%m/%y")
mine_price["date"] = pd.to_datetime(mine_price["date"], format="%d/%m/%y")

# change percentages to numeric
cu_ex_1["cme_perc_chg"] = (
    cu_ex_1["cme_perc_chg"].str.replace("%", "").astype(float) / 100
)

# cu_ex_3["lme_value"] = cu_ex_3["lme_value"] * 10

# Merge all dataframes
df2 = pd.merge(cu_ex_1, cu_ex_2, on="date", how="outer")
df2 = pd.merge(df2, mine_price, on="date", how="outer")
df2 = df2.sort_values(by="date")

df2.columns

# %%
df = pd.merge(df,df2,on="date", how="outer")
cols = df.columns.difference(["date"])
df[cols] = df[cols].astype(float)

# %%
# # Reusable function to plot all the data (for visual inspection)


# def plot_data(
#     data: pd,
#     figsize: tuple = (14, 3),
#     title: str = "Copper Data",
#     stocks: list = None,
#     exchanges: list = None,
# ):
#     fig, ax = plt.subplots(1, 4, figsize=figsize)

#     exchanges = exchanges or ["cme", "shfe", "lme"]
#     stocks = stocks or ["rio", "sqm", "fcx", "bhp", "nem", "scco", "hbm", "ero", "vale"]

#     # if "lme" in exchanges:
#     #     ax[2].plot(data["date"], data["lme_CU_close"], label="LME Copper Value")
#     #     ax[2].set_title("LME Copper Exchange")
#     #     exchanges.remove("lme")

#     for i, exch in enumerate(exchanges):
#         ax[i].plot(data["date"], data[f"{exch}_high"], label=f"high")
#         ax[i].plot(data["date"], data[f"{exch}_low"], label=f"low")
#         ax[i].plot(data["date"], data[f"{exch}_open"], label=f"open")
#         ax[i].plot(data["date"], data[f"{exch}_close"], label=f"close")
#         ax[i].set_title(f"{exch.upper()} Copper Exchange")

#     for i, stock in enumerate(stocks):
#         ax[3].plot(
#             data["date"], data[f"{stock}_adj_close"], label=f"{stock.upper()} Close"
#         )

#     ax[3].set_title("Mining Companies")

#     plt.tight_layout()
#     plt.suptitle(title)
#     plt.show()


# plot_data(df, title="Copper Data")

# %%
# Now lets reindex the data and interpolate the missing values

date_range = pd.date_range(start=df["date"].min(), end="2024-07-31", freq="D")
df_reindexed = (
    df.set_index("date")
    .reindex(date_range)
    .reset_index()
    .rename(columns={"index": "date"})
)
df_interpolated = df_reindexed.interpolate(method='linear')
df_final = df_interpolated.ffill().bfill()

# plot_data(df_final, title="Copper Data", figsize=(14, 3), stocks=["rio", "sqm"])
df_final.columns

# %%
# Writing Data to CSV File: 
df_final.to_csv("copper_data.csv")

# %%
# Regression for relationship between CME copper and FCX stock price - trained on 98% of the data to predict the last 20%
split = int(0.80 * len(df_final))
train = df_final[:split]
test = df_final[split:] 


# %%
def linear_regression(train, test, x_col, y_col):
    
    X = sm.add_constant(train[x_col])
    model = sm.OLS(train[y_col], X).fit()
    print(model.summary())
    return model.predict(sm.add_constant(test[x_col]))


def plot_linear_regression(train, test, x_col, y_col, y_pred):
    x_lab = x_col.split("_")[0].upper()
    y_lab = y_col.split("_")[0].upper()
    plt.plot(df_final["date"], df_final[x_col], label=f"{x_lab} Share price")
    plt.plot(train["date"], train[y_col], label=f"{y_lab} Share Price ($/lb)")
    plt.plot(test["date"], test[y_col], label=f"Actual {y_lab} Price ($/lb)")
    plt.plot(test["date"], y_pred, label=f"{y_lab} Price ($/lb) Prediction")
    plt.xlabel("Date")  # Add x-axis label
    plt.ylabel("Price in USD")  # Add y-axis label

    # plt.plot(test[x_col], y_pred, label="Prediction")
    plt.legend()
    plt.show()
    


def calculate_mean_absolute_percentage_error(data, y_col, y_pred):
    return (
        ((y_pred - data[y_col]) / data[y_col])
        .abs()
        .mean()
        .item()
    )


# %%

y_pred = linear_regression(train, test, "fcx_adj_close", "cme_close")

plot_linear_regression(train, test, "fcx_adj_close", "cme_close", y_pred)

calculate_mean_absolute_percentage_error(test, "cme_close", y_pred)

# %%
X = sm.add_constant(train["cme_close"])
model = sm.OLS(train["bhp_adj_close"], X).fit()
print(model.summary())
f_pred = model.predict(sm.add_constant(train["cme_close"]))

plt.figure(figsize=(12, 6))

plt.semilogy(df_final["date"], df_final["cme_close"], label=f"CME Copper Price")
plt.semilogy(train["date"], train["bhp_adj_close"], label=f"BHP Stock Price")
plt.semilogy(test["date"], test["bhp_adj_close"], label=f"Actual BHP Stock Price")
plt.semilogy(test["date"], y_pred, label=f"BHP Stock Price Prediction")
plt.semilogy(train["date"], f_pred, label=f"3 BHP Stock Price")

plt.xlabel("Date")
plt.ylabel("Price (log scale)")
plt.title(f"BHP Stock Price vs CME Copper Price")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

calculate_mean_absolute_percentage_error(test, "bhp_adj_close", y_pred)

# %%
summary = []
for ex_id in ["shfe_close", "cme_close", "LME 3M CU_close"]:
  for stock_id in ["scco_adj_close", "vale_adj_close", "Zijin_close",	"Jiangxi_close", "SFR_close", "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]:
       
      y_pred = linear_regression(train, test, ex_id, stock_id)
      mape = calculate_mean_absolute_percentage_error(test, stock_id, y_pred)
      summary.append({"ex_id": ex_id, "stock_id": stock_id, "mape": mape})

reg_df_1 = pd.DataFrame(summary)
reg_df_1.to_clipboard()

# %%
#multi variable linear regression 
def mv_regression(train, test, x_cols, y_col):
    # Reset indices to avoid any potential indexing issues
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Print column names and check for consistency


    # Add constant and check shapes
    X = sm.add_constant(train[x_cols])

    # Ensure no NaNs

    model = sm.OLS(train[y_col], X).fit()
    print(model.summary())

    # Prepare test data
    x_test = sm.add_constant(test[x_cols])


      # Check if X and x_test have the same number of columns
    if X.shape[1] != x_test.shape[1]:
        raise ValueError("Mismatch in the number of columns between train and test data")

    predictions = model.predict(x_test)
    print(f"Predictions shape: {predictions.shape}")

    return model, predictions


# %%
summary = []
models = []
Xs = ["SHFE price_close", "LME 3M CU_close", "cme_close"]

for stock_id in ["scco_adj_close", "vale_adj_close", "Zijin_close", "Jiangxi_close", "SFR_close", "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]:
    

    model, y_pred = mv_regression(train, test, Xs, stock_id)
    mape = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).abs().mean().item()
    mpe = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).mean().item()


    omnibus_test = sm.stats.omni_normtest(model.resid)
    jb_test = sm.stats.jarque_bera(model.resid)

    summary.append({
        "stock_id": stock_id,
        "MAPE": mape,
        'R2': model.rsquared,
        'P_value': model.f_pvalue,
        'F_stat': model.fvalue,
        'Skew': model.resid.skew(),
        'Kurtosis': model.resid.kurtosis(),
        'Durbin_Watson': sm.stats.durbin_watson(model.resid),
        'Omnibus': omnibus_test[0],  # Omnibus test statistic
        'Omnibus_pvalue': omnibus_test[1],  # p-value for the Omnibus test
        'Jarque_Bera': jb_test[0],  # Jarque-Bera test statistic
        'Jarque_Bera_pvalue': jb_test[1],  # p-value for the Jarque-Bera test
        'MPE': mpe
    })

reg_df_1 = pd.DataFrame(summary)
reg_df_1.to_clipboard()
reg_df_1 


# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

def mv_regression(train, test, Xs, y):
    X_train = sm.add_constant(train[Xs])
    y_train = train[y]
    
    model = sm.OLS(y_train, X_train).fit()
    
    X_test = sm.add_constant(test[Xs])
    y_pred = model.predict(X_test)
    
    return model, y_pred

# List of stock IDs
stock_ids = ["scco_adj_close", "vale_adj_close", "Zijin_close", "Jiangxi_close", "SFR_close", "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]

# Aggregate the mining companies
train['aggregated_mining'] = train[stock_ids].mean(axis=1)
test['aggregated_mining'] = test[stock_ids].mean(axis=1)

Xs = ["SHFE price_close", "LME 3M CU_close", "cme_close"]

# Perform regression on aggregated data
model, y_pred = mv_regression(train, test, Xs, 'aggregated_mining')

# Calculate metrics
mape = np.abs((test['aggregated_mining'].to_numpy() - y_pred.to_numpy()) / test['aggregated_mining']).mean()
mpe = ((test['aggregated_mining'].to_numpy() - y_pred.to_numpy()) / test['aggregated_mining']).mean()

omnibus_test = sm.stats.omni_normtest(model.resid)
jb_test = sm.stats.jarque_bera(model.resid)

# Create summary dictionary
summary = {
    "MAPE": mape,
    'R2': model.rsquared,
    'P_value': model.f_pvalue,
    'F_stat': model.fvalue,
    'Skew': model.resid.skew(),
    'Kurtosis': model.resid.kurtosis(),
    'Durbin_Watson': sm.stats.durbin_watson(model.resid),
    'Omnibus': omnibus_test[0],  # Omnibus test statistic
    'Omnibus_pvalue': omnibus_test[1],  # p-value for the Omnibus test
    'Jarque_Bera': jb_test[0],  # Jarque-Bera test statistic
    'Jarque_Bera_pvalue': jb_test[1],  # p-value for the Jarque-Bera test
    'MPE': mpe
}

# Create DataFrame from summary
reg_df = pd.DataFrame([summary])

# Copy to clipboard and display results
reg_df.to_clipboard()
print(reg_df)

# Display model summary
print(model.summary())

# %%
y_pred = linear_regression(train, test, "shfe_close", "ANTO_close")

plot_linear_regression(train, test, "shfe_close", "ANTO_close", y_pred)

calculate_mean_absolute_percentage_error(test, "ANTO_close", y_pred)

#new

# %%
summary = []
models = []
Xs = ["scco_adj_close", "vale_adj_close", "Zijin_close", "Jiangxi_close", "SFR_close", "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]

for stock_id in ["LME 3M CU_close", "cme_close", "shfe_close"]:
    

    model, y_pred = mv_regression(train, test, Xs, stock_id)
    mape = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).abs().mean().item()
    mpe = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).mean().item()


    omnibus_test = sm.stats.omni_normtest(model.resid)
    jb_test = sm.stats.jarque_bera(model.resid)

    summary.append({
        "stock_id": stock_id,
        "MAPE": mape,
        'MPE': mpe
    })

reg_df_1 = pd.DataFrame(summary)
reg_df_1.to_clipboard()
reg_df_1 

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def combine_data(df, exchange_columns, stock_columns):
    # Combine exchange data
    exchanges = df[exchange_columns].sum(axis=1)
    
    # Combine stock data
    stocks = df[stock_columns].mean(axis=1)
    
    return pd.DataFrame({'exchanges': exchanges, 'stocks': stocks})

def perform_regression(train, test):
    model = LinearRegression()
    model.fit(train[['exchanges']], train['stocks'])
    
    y_pred_train = model.predict(train[['exchanges']])
    y_pred_test = model.predict(test[['exchanges']])
    
    mape_train = np.mean(np.abs((train['stocks'] - y_pred_train) / train['stocks'])) * 100
    mape_test = np.mean(np.abs((test['stocks'] - y_pred_test) / test['stocks'])) * 100
    
    return {
        'Intercept': model.intercept_,
        'Coefficient': model.coef_[0],
        'MAPE (Train)': mape_train,
        'MAPE (Test)': mape_test
    }

# Define your column names
exchange_columns = ["cme_volume", "shfe_volume", "LME 3M CU_volume"]
stock_columns = ["scco_adj_close", "vale_adj_close", "Zijin_close", "Jiangxi_close", "SFR_close", 
                 "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", 
                 "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", 
                 "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", 
                 "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]

# Combine data for train and test sets
train_combined = combine_data(train, exchange_columns, stock_columns)
test_combined = combine_data(test, exchange_columns, stock_columns)

# Perform regression and get results
results = perform_regression(train_combined, test_combined)

# Create and display results table
results_table = pd.DataFrame([results])
print(results_table)

# %%

y_pred = linear_regression(train, test, "cme_close", "CHINADAYE_close")

plot_linear_regression(train, test, "cme_close", "CHINADAYE_close", y_pred)

calculate_mean_absolute_percentage_error(test, "CHINADAYE_close", y_pred)

# %%
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Assuming train and test dataframes are already defined and mv_regression is available

summary = []
models = []
betas = []
Xs = [
    "scco_adj_close",
    "vale_adj_close",
    "Zijin_close",
    "Jiangxi_close",
    "SFR_close",
    "Glen_close",
    "rio_adj_close",
    "fcx_adj_close",
    "bhp_adj_close",
    "nem_adj_close",
    "hbm_adj_close",
    "ero_adj_close",
    "CMOC_close",
    "BAO_close",
    "IVAN_close",
    "FQM_close",
    "ANTO_close",
    "CAPS_close",
    "LUN_close",
    "MINS_close",
    "MMG_close",
    "CHINADAYE_close",
    "Jinchuan_close",
    "CAML_price",
    "Taseko_close",
]

for stock_id in ["SHFE price_close", "LME 3M CU_close", "cme_close"]:
    model, y_pred = mv_regression(train, test, Xs, stock_id)
    mape = (
        ((test[stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id])
        .abs()
        .mean()
        .item()
    )
    mpe = (
        ((test[stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).mean().item()
    )

    omnibus_test = sm.stats.omni_normtest(model.resid)
    jb_test = sm.stats.jarque_bera(model.resid)

    summary.append(
        {
            "stock_id": stock_id,
            "MAPE": mape,
            "R2": model.rsquared,
            "P_value": model.f_pvalue,
            "F_stat": model.fvalue,
            "Skew": model.resid.skew(),
            "Kurtosis": model.resid.kurtosis(),
            "Durbin_Watson": sm.stats.durbin_watson(model.resid),
            "Omnibus": omnibus_test[0],  # Omnibus test statistic
            "Omnibus_pvalue": omnibus_test[1],  # p-value for the Omnibus test
            "Jarque_Bera": jb_test[0],  # Jarque-Bera test statistic
            "Jarque_Bera_pvalue": jb_test[1],  # p-value for the Jarque-Bera test
            "MPE": mpe,
        }
    )

    # Now we calculate the standardised beta coefficients
    X_orig = train[Xs]
    X_scaled = (X_orig - X_orig.mean()) / X_orig.std()
    X_scaled = sm.add_constant(X_scaled)
    y_scaled = (train[stock_id] - train[stock_id].mean()) / train[stock_id].std()

    # Now we run the regression
    model_scaled = sm.OLS(y_scaled, X_scaled).fit()

    # Now we can extract the beta coefficients
    beta = model_scaled.params
    betas.append(
        {"stock_id": stock_id, **{col: beta[col].item() for col in X_scaled.columns}}
    )

reg_df_1 = pd.DataFrame(summary)

# Print the summary DataFrame
print("Summary of Regression Results:")
print(reg_df_1.to_string(index=False))

# Now we can calculate the standardised beta coefficients
beta_df = pd.DataFrame(betas)

# Print the beta coefficients DataFrame
print("\nStandardised Beta Coefficients:")
print(beta_df.to_string(index=False))

# Plot beta_df as bar graph with columns as x axis and beta values as y axis, each stock as a different color
# Other plot absolute beta values
# Change series names with fn lambda x: x.split("_")[0]

beta_df_g = beta_df.copy()
beta_df_g["stock_id"] = beta_df_g["stock_id"].apply(lambda x: x.split("_")[0])
beta_df_g = beta_df_g.set_index("stock_id")
beta_df_g.columns = beta_df_g.columns.map(lambda x: x.split("_")[0])
beta_df_g = beta_df_g.T

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Adjusted figsize for better display

# Plot the beta coefficients
beta_df_g.plot(kind="bar", ax=axs[0])
axs[0].set_title("Beta Coefficients")
axs[0].set_ylabel("Beta Coefficient")
axs[0].set_xlabel("Stocks")
axs[0].legend(title="Exchange:", bbox_to_anchor=(1, 1))
axs[0].grid(True, which="both", ls="-", alpha=0.5)

axs[0].set_xticklabels([label.get_text().upper() for label in axs[0].get_xticklabels()], rotation=45)

# Convert x-tick labels to uppercase
axs[0].set_xticklabels([label.get_text().upper() for label in axs[0].get_xticklabels()])

# Plot the absolute beta coefficients
beta_df_g.apply(lambda x: x.abs()).plot(kind="bar", ax=axs[1])
axs[1].set_title("Absolute Beta Coefficients")
axs[1].set_ylabel("Absolute Beta Coefficient")
axs[1].set_xlabel("Stocks")
axs[1].legend(title="Exchange:", bbox_to_anchor=(1, 1))
axs[1].grid(True, which="both", ls="-", alpha=0.5)

axs[1].set_xticklabels([label.get_text().upper() for label in axs[1].get_xticklabels()], rotation=45)

# Convert x-tick labels to uppercase
axs[1].set_xticklabels([label.get_text().upper() for label in axs[1].get_xticklabels()])

plt.tight_layout()
plt.show()



# %%
summary = []
models = []
Xs = ["scco_adj_close", "vale_adj_close", "Zijin_close", "Jiangxi_close", "SFR_close", "Glen_close", "rio_adj_close", "fcx_adj_close", "bhp_adj_close", "nem_adj_close", "hbm_adj_close", "ero_adj_close", "CMOC_close", "BAO_close", "IVAN_close", "FQM_close", "ANTO_close", "CAPS_close", "LUN_close", "MINS_close", "MMG_close", "CHINADAYE_close", "Jinchuan_close", "CAML_price", "Taseko_close"]

for stock_id in ["SHFE price_close", "LME 3M CU_close", "cme_close"]:
    

    model, y_pred = mv_regression(train, test, Xs, stock_id)
    mape = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).abs().mean().item()
    mpe = ((test [stock_id].to_numpy() - y_pred.to_numpy()) / test[stock_id]).mean().item()


    omnibus_test = sm.stats.omni_normtest(model.resid)
    jb_test = sm.stats.jarque_bera(model.resid)

    summary.append({
        "stock_id": stock_id,
        "MAPE": mape,
        'R2': model.rsquared,
        'P_value': model.f_pvalue,
        'F_stat': model.fvalue,
        'Skew': model.resid.skew(),
        'Kurtosis': model.resid.kurtosis(),
        'Durbin_Watson': sm.stats.durbin_watson(model.resid),
        'Omnibus': omnibus_test[0],  # Omnibus test statistic
        'Omnibus_pvalue': omnibus_test[1],  # p-value for the Omnibus test
        'Jarque_Bera': jb_test[0],  # Jarque-Bera test statistic
        'Jarque_Bera_pvalue': jb_test[1],  # p-value for the Jarque-Bera test
        'MPE': mpe
    })

reg_df_1 = pd.DataFrame(summary)
reg_df_1.to_clipboard()
reg_df_1 


# %%
import matplotlib.pyplot as plt
import statsmodels.api as sm

def linear_regression(train, test, x_col, y_col):
    X = sm.add_constant(train[x_col])
    model = sm.OLS(train[y_col], X).fit()
    print(model.summary())
    return model.predict(sm.add_constant(test[x_col]))

def plot_linear_regression(train, test, x_col, y_col, y_pred):
    x_lab = x_col.split("_")[0].upper()
    y_lab = y_col.split("_")[0].upper()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_final["date"], df_final[x_col], label=f"{x_lab} Share price")
    ax.plot(train["date"], train[y_col], label=f"{y_lab} Share Price ($/tn)")
    ax.plot(test["date"], test[y_col], label=f"Actual {y_lab} Price ($/tn)")
    ax.plot(test["date"], y_pred, label=f"{y_lab} Price ($/tn) Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price in USD")
    ax.set_yscale("log")
    ax.legend()
    plt.show()

def calculate_mean_absolute_percentage_error(data, y_col, y_pred):
    return (
        ((y_pred - data[y_col]) / data[y_col])
        .abs()
        .mean()
        .item()
    )

y_pred = linear_regression(train, test, "rio_adj_close", "shfe_close")
plot_linear_regression(train, test, "rio_adj_close", "shfe_close", y_pred)
calculate_mean_absolute_percentage_error(test, "shfe_close", y_pred)


