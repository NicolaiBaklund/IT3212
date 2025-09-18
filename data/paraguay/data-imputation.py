import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm


# Algoritmo S-ESD y S-H-ESD
def ESD_test(type_algorithm, data, alpha, max_outliers):
    indexes = []
    for iterations in range(max_outliers):

        # Calculate the Grubbs Statistic value
        if type_algorithm == 'sh':
            mad = stats.median_abs_deviation(data)
            median_data = np.median(data)
            abs_diff = abs(data - median_data)
            max_abs_diff = max(abs_diff)
            max_ind = abs_diff.idxmax()
            Gstat = max_abs_diff / mad

        elif type_algorithm == 's':
            std_dev = np.std(data)
            avg_data = np.mean(data)
            abs_diff = abs(data - avg_data)
            max_abs_diff = max(abs_diff)
            max_ind = abs_diff.idxmax()
            Gstat = max_abs_diff / std_dev

        # Calculate the Grubbs Critical value
        N = len(data)
        t_disp = stats.t.ppf(1 - alpha / (2 * N), N - 2)
        num = (N - 1) * np.sqrt(np.square(t_disp))
        den = np.sqrt(N) * np.sqrt(N - 2 + np.square(t_disp))
        critical_value = num / den

        # Check G values
        if Gstat > critical_value:
            indexes.append(max_ind)
            data = data.drop(max_ind)

        else:
            i = 1

    return indexes, data


df = pd.read_csv('electricity-consumption-raw.csv', header=None, skiprows=1, names=["datetime", "substation", "feeder", "consumption"])
substations = np.unique(df["substation"])
feeders = np.unique(df["feeder"])
df.set_index(["substation", "feeder", "datetime"], inplace=True)
df.sort_index(inplace=True)

# Diccionario de los alimentadores
feed_dict = {}
for sub in substations:
    feed_dict[sub] = list(np.unique(df.loc[sub].index.get_level_values(0)))

# DataFrame de los alimentadores
df_feeders = pd.DataFrame(index=pd.date_range(start='1/1/2017 00:00', end='31/12/2020 23:00', freq='H'))

for k in feed_dict.keys():
    for i in feed_dict[k]:
        df_feeders = pd.concat([df_feeders, df.loc[k].loc[i]["consumption"]], axis=1)
        df_feeders.rename(columns={'consumption': i}, inplace=True)

timeSeries = df_feeders.copy()

feeders_filter = pd.DataFrame(index=pd.date_range(start='1/1/2017 00:00', end='31/12/2020 23:00', freq='H'))
feeders_filter_dropna = pd.DataFrame(index=pd.date_range(start='1/1/2017 00:00', end='31/12/2020 23:00', freq='H'))

outliers_index = []

for ts in tqdm(timeSeries.columns):
    indexes_nan = timeSeries[timeSeries[ts].isna()].index
    timeSeries[ts] = timeSeries[ts].resample('H').interpolate(method='linear')
    timeSeries[ts] = timeSeries[ts].iloc[::-1].interpolate(method='linear').iloc[::-1]

    timeSeries["BC"], lambda_op = stats.boxcox(timeSeries[ts])
    res = STL(timeSeries["BC"], trend=37, period=24, seasonal=147, robust=True).fit()

    days = 14
    piecewise_trend = timeSeries["BC"].copy()
    K = int(len(timeSeries["BC"].values) / (days * 24)) + 1
    for k in range(K):
        piecewise_trend.values[k * (days * 24):(k + 1) * (days * 24)] = np.median(timeSeries["BC"].values[k * (days * 24):(k + 1) * (days * 24)])

    new_residual = timeSeries["BC"] - res.seasonal - piecewise_trend
    index, data = ESD_test('sh', new_residual, 0.05, 5000)

    index.sort()
    seasonally_adjusted = piecewise_trend + new_residual

    for i in index:
        seasonally_adjusted.loc[i] = float(np.nan)

    seasonally_adjusted = seasonally_adjusted.resample('H').interpolate(method='linear')
    seasonally_adjusted = seasonally_adjusted.iloc[::-1].resample('H').interpolate(method='linear').iloc[::-1]

    time_series_clean = sp.special.inv_boxcox(seasonally_adjusted + res.seasonal, lambda_op)

    for i in indexes_nan:
        time_series_clean.loc[i] = float(np.nan)

    feeders_filter_dropna[ts] = time_series_clean.copy()

    # Rellenado por promedio

    df_fill = pd.DataFrame(time_series_clean, columns=["T"])
    df_fill["NAN"] = df_fill.isna()

    weeks = 24 * 7
    for steps in range(-6, 7):
        if steps != 0:
            df_fill["T" + str(steps)] = df_fill["T"].shift(periods=weeks * steps)

    df_fill["N"] = 12 - df_fill[df_fill.columns[2:]].isna().sum(axis=1)
    df_fill[df_fill.columns[2:]] = df_fill[df_fill.columns[2:]].fillna(0)

    df_fill["Mean"] = df_fill[df_fill.columns[2:]].sum(axis=1)
    df_fill.loc[df_fill["Mean"] != 0, "Mean"] = df_fill["Mean"] / df_fill["N"]
    df_fill.loc[df_fill["Mean"] == 0, "Mean"] = float(np.nan)

    df_fill.loc[df_fill["NAN"] == True, "T"] = df_fill["Mean"]

    feeders_filter[ts] = df_fill["T"]

    outliers_index.append([ts, index])
    seasonal_adjusted = None
    time_series_clean = None

print("Finished!")

columns = ["index_outlier", "substation", "feeder"]
df_aux = pd.DataFrame(columns=columns)

for j in list(feed_dict.items()):
    for k in j[1]:
        df = pd.DataFrame(columns=columns)
        for feed in outliers_index:
            if feed[0] == k:
                df["index_outlier"] = feed[1]
                break

        df["substation"] = j[0]
        df["feeder"] = k
        df_aux = df_aux.append(df, ignore_index=False)

filtered = feeders_filter.copy()

# Eliminacion de los dias que aun poseen datos faltantes (Se coloca como "nan" los valores de ese dia)
nan_days = []
for label in filtered.columns:
    one_day = []
    hours = 0
    for date in pd.date_range(start='1/1/2017', end='31/12/2020'):
        day_consumption = filtered[label][str(date.date())].values

        if np.isnan(day_consumption).sum() > 0:
            nan_days.append(date.date())
            one_day.append(str(date.date()))
            hours += np.isnan(day_consumption).sum()

nan_days = list(dict.fromkeys(nan_days))
for label in filtered.columns:
    for date in nan_days:
        filtered.loc[filtered.index.date == date, label] = float(np.nan)

outputList = []
for tis, row in filtered.iterrows():
    s = tis.isoformat()
    for j, e in row.items():
        outputList.append([s[:19], j[0], j, e])

df_transpose = pd.DataFrame(outputList, columns=['datetime', 'substation', 'feeder', 'consumption'])
df_transpose = df_transpose.sort_values(by=['feeder','datetime'])
df_transpose.to_csv('electricity-consumption-processed.csv', index=False)
