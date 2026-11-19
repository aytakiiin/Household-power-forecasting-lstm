"""Build data/daily_aggregated.csv from the raw UCI file.
Usage: python src/prepare_data.py path/to/household_power_consumption.txt
"""
import sys, pandas as pd
raw = sys.argv[1] if len(sys.argv) > 1 else 'household_power_consumption.txt'
df = pd.read_csv(raw, sep=';', na_values=['?', ''], low_memory=False)
df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(columns=['Date', 'Time']).set_index('dt').sort_index()
df['gap_wh'] = df['Global_active_power'] * 1000 / 60                       # kW (1-min mean) -> Wh per minute
df['rest_wh'] = df['gap_wh'] - df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum(axis=1)
d = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']].resample('D').mean()
d['kwh_adj'] = d['Global_active_power'] * 24                              # coverage-adjusted daily kWh
d['cov'] = df['Global_active_power'].resample('D').count() / 1440          # share of minutes with data
for i in (1, 2, 3): d[f'sm{i}'] = df[f'Sub_metering_{i}'].resample('D').sum() / 1000
d['sm_rest'] = df['rest_wh'].resample('D').sum() / 1000
d = d[d['cov'] > 0]
d.to_csv('data/daily_aggregated.csv'); print(d.shape)
