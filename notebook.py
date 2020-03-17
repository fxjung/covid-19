# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline

import sys
import os
import shutil
import datetime
import requests
import urllib.request

import itertools as it
import functools as ft

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from pathlib import Path
from scipy.optimize import curve_fit

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.fontsize"] = 16
mpl.rcParams["lines.markersize"] = 15

mpl.rc("text", usetex=True)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

sns.set(color_codes=True, font_scale=1.5)
sns.set_style("ticks",)

evf = lambda S, f, **arg: (S, f(S, **arg))

os.getcwd()
# -

# # Preprocessing
# ## Downloading Data

# +
data = Path("data")
data.mkdir(exist_ok=True)
assert data.exists()

# day_to_download = pd.Timestamp('2020-03-16')
day_to_download = pd.Timestamp.now()

xlsname = f"COVID-19-geographic-disbtribution-worldwide-{day_to_download.strftime('%Y-%m-%d')}.xls"
url = "http://www.ecdc.europa.eu/sites/default/files/documents/" + xlsname
print(f"Trying to download from {url}")
try:
    with urllib.request.urlopen(url) as response, (data / xlsname).open(
        "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)
except urllib.request.HTTPError as e:
    if "404" in str(e):
        print("ERROR: Download failed (404). Probably today's data is not there yet.")
    else:
        raise

sorted_paths = sorted(data.glob("*.xls"), key=os.path.getmtime)
print("\navailable files:\n" + "\n".join(map(str, sorted_paths)) + "\n")

xlspath = sorted_paths[-1]
print(f"selected file: {xlspath}")

# +
df = pd.read_excel(xlspath, parse_dates=["DateRep"])
df.loc[:, "CountryExp"] = df["CountryExp"].apply(lambda x: x[0].upper() + x[1:])
df.set_index(["CountryExp", "DateRep"], inplace=True)
df.sort_values(["CountryExp", "DateRep"], inplace=True)

df.loc[:, "total_cases"] = df.groupby("CountryExp")["NewConfCases"].cumsum()
df.loc[:, "total_deaths"] = df.groupby("CountryExp")["NewDeaths"].cumsum()
df["growth_factor_cases"] = df.groupby("CountryExp", group_keys=False).apply(
    lambda df: df["NewConfCases"] / df["NewConfCases"].shift(1)
)
df["growth_factor_deaths"] = df.groupby("CountryExp", group_keys=False).apply(
    lambda df: df["NewDeaths"] / df["NewDeaths"].shift(1)
)
# -

# # Plots by Country

# +
tasks = [
    {
        "countries": [
            "Germany",
            "Italy",
            "South Korea",
            "United States of America",
            "Spain",
            "France",
            "Austria",
            "Switzerland",
        ],
    },
    {"countries": ["Germany", "Italy", "South Korea",],},
    {
        "countries": ["Germany",],
        "interpolate": True,
        "extrapolate": True,
        "show_events": True,
    },
    {
        "countries": ["Italy",],
        "interpolate": True,
        "extrapolate": True,
        "show_events": True,
    },
]
style = "X:"
alpha = 1
window_size = 10
projected_delta = pd.Timedelta(days=12)
events = {
    "Germany": [
        ("2020-03-10", "events $>$ 1000 canceled"),
        ("2020-03-12", "Merkel address"),
        ("2020-03-15", "borders shut"),
        ("2020-03-17", "public shutdown"),
    ],
    "Italy": [("2020-03-9", "lockdown"),],
}


for task in tasks:
    countries = task["countries"]
    extrapolate = task.get("extrapolate", False)
    interpolate = task.get("interpolate", False)
    show_events = task.get("show_events", False)
    if extrapolate:
        event_countries = set(countries).intersection(events)
        if not event_countries:
            xmax = pd.Timestamp.now() + projected_delta
        else:
            xmax = (
                max(
                    pd.Timestamp(epoch)
                    for event_country in event_countries
                    for epoch, _ in events[event_country]
                )
                + projected_delta
                + pd.Timedelta(days=1)
            )
    else:
        xmax = pd.Timestamp.now()
    fig, axs = plt.subplots(4, 2, figsize=(2 * 8, 4 * (8 if event_countries else 6)))

    for country in countries:
        data = df.loc[country]

        axs[0][0].plot(data["total_cases"], style, label=country, alpha=alpha)
        axs[0][1].plot(data["total_deaths"], style, label=country, alpha=alpha)

        axs[1][0].plot(data["total_cases"], style, label=country, alpha=alpha)
        axs[1][1].plot(data["total_deaths"], style, label=country, alpha=alpha)

        (gfc,) = axs[2][0].plot(data["growth_factor_cases"], style, alpha=alpha)
        (gfd,) = axs[2][1].plot(data["growth_factor_deaths"], style, alpha=alpha)

        axs[2][0].plot(
            data["growth_factor_cases"].rolling(window_size, center=True).mean(),
            "-",
            lw=3,
            label=country,
            color=gfc.get_color(),
        )
        axs[2][1].plot(
            data["growth_factor_deaths"].rolling(window_size, center=True).mean(),
            "-",
            lw=3,
            label=country,
            color=gfd.get_color(),
        )

        axs[3][0].plot(data["NewConfCases"], style, label=country, alpha=alpha)
        axs[3][1].plot(data["NewDeaths"], style, label=country, alpha=alpha)

        if interpolate or extrapolate:
            model = lambda t, a, b: a * np.exp(b * t)
            t = ((data.index - data.index[0]) / pd.Timedelta(days=1)).to_numpy()
            c = data["total_cases"].to_numpy()
            d = data["total_deaths"].to_numpy()

            c_f = curve_fit(model, t, c, p0=(1e-5, 1e-1))
            d_f = curve_fit(model, t, d, p0=(1e-5, 1e-1))

            print(c_f[0])
            print(d_f[0])

            dt_t = pd.date_range(data.index[0], xmax)
            i_t = ((dt_t - dt_t[0]) / pd.Timedelta(days=1)).to_numpy()
            for ax in axs[:2, 0]:
                ax.plot(
                    dt_t, model(i_t, a=c_f[0][0], b=c_f[0][1]), ":", color="C1"
                )  # gfc.get_color())

            for ax in axs[:2, 1]:
                ax.plot(
                    dt_t, model(i_t, a=d_f[0][0], b=d_f[0][1]), ":", color="C1"
                )  # gfd.get_color())

    for ax in axs[:2, 0]:
        ax.set_ylabel("total cases")

    for ax in axs[:2, 1]:
        ax.set_ylabel("total deaths")

    axs[2][0].set_ylabel(
        r"daily growth factor cases $\frac{\Delta n^\mathrm{c}_i}{\Delta n^\mathrm{c}_{i-1}}$"
    )
    axs[2][1].set_ylabel(
        r"daily growth factor deaths $\frac{\Delta n^\mathrm{d}_i}{\Delta n^\mathrm{d}_{i-1}}$"
    )

    axs[3][0].set_ylabel("new cases per day")
    axs[3][1].set_ylabel("new deaths per day")

    axs[2][0].set_ylim(0, 5)
    axs[2][1].set_ylim(0, 5)

    for ax in axs[0]:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 4))

    for ax in axs[1]:
        ax.set_yscale("log")

    for ax_ in axs:
        for ax in ax_:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=15, maxticks=52))
            ax.xaxis.set_major_formatter(formatter)

            ax.legend(loc=2)
            ax.grid(True)
            ax.set_ylim(1, ax.get_ylim()[1])

            if show_events:
                ylim = ax.get_ylim()
                if ax.get_yscale() == "log":
                    ypos = np.exp(
                        (np.log(ylim[1]) + (np.log(ylim[1]) - np.log(ylim[0])) * 0.02)
                    )
                else:
                    ypos = ylim[1] + (ylim[1] - ylim[0]) * 0.02
                for country in countries:
                    for epoch, text in events.get(country, []):
                        ts = pd.Timestamp(epoch)
                        projected_ts = ts + projected_delta
                        ax.axvline(ts, c="r", ls="-")
                        ax.axvline(projected_ts, c="r", ls=":")
                        ax.text(
                            ts,
                            ypos,
                            text,
                            ha="center",
                            va="bottom",
                            fontsize=15,
                            rotation=90,
                        )
                        ax.text(
                            projected_ts,
                            ypos,
                            text,
                            ha="center",
                            va="bottom",
                            fontsize=15,
                            rotation=90,
                            color="grey",
                        )
            ax.set_xlim(pd.Timestamp("2020-02-25"), xmax)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(
            f'linlog{"_interpol" if interpolate else ""}{"_extrapol" if extrapolate else ""}{"_events" if show_events else ""}_{"_".join(df.loc[country]["GeoId"][0] for country in countries)}.{ext}',
            dpi=200,
        )
# -

# # Plots China vs World

# +
data_world = df.drop('China').groupby("DateRep").sum()
data_world.loc[:, "total_cases"] = data_world["NewConfCases"].cumsum()
data_world.loc[:, "total_deaths"] = data_world["NewDeaths"].cumsum()
data_world["growth_factor_cases"] = data_world["NewConfCases"] / data_world["NewConfCases"].shift(1)
data_world["growth_factor_deaths"] = data_world["NewDeaths"] / data_world["NewDeaths"].shift(1)

data_china = df.loc['China'].copy()
data_china.loc[:, "total_cases"] = data_china["NewConfCases"].cumsum()
data_china.loc[:, "total_deaths"] = data_china["NewDeaths"].cumsum()
data_china["growth_factor_cases"] = data_china["NewConfCases"] / data_china["NewConfCases"].shift(1)
data_china["growth_factor_deaths"] = data_china["NewDeaths"] / data_china["NewDeaths"].shift(1)

fig, axs = plt.subplots(4, 2, figsize=(2 * 8, 4 * 6))
style = "X:"
alpha = 1
window_size = 10

for data, country in zip([data_world, data_china], ["rest", "china"]):
    axs[0][0].plot(data["total_cases"], style, label=country, alpha=alpha)
    axs[0][1].plot(data["total_deaths"], style, label=country, alpha=alpha)

    axs[1][0].plot(data["total_cases"], style, label=country, alpha=alpha)
    axs[1][1].plot(data["total_deaths"], style, label=country, alpha=alpha)

    (gfc,) = axs[2][0].plot(data["growth_factor_cases"], style, alpha=alpha)
    (gfd,) = axs[2][1].plot(data["growth_factor_deaths"], style, alpha=alpha)

    axs[2][0].plot(
        data["growth_factor_cases"].rolling(window_size, center=True).mean(),
        "-",
        lw=3,
        label=country,
        color=gfc.get_color(),
    )
    axs[2][1].plot(
        data["growth_factor_deaths"].rolling(window_size, center=True).mean(),
        "-",
        lw=3,
        label=country,
        color=gfd.get_color(),
    )

    axs[3][0].plot(data["NewConfCases"], style, label=country, alpha=alpha)
    axs[3][1].plot(data["NewDeaths"], style, label=country, alpha=alpha)


for ax in axs[:2, 0]:
    ax.set_ylabel("total cases")

for ax in axs[:2, 1]:
    ax.set_ylabel("total deaths")

axs[2][0].set_ylabel(
    r"daily growth factor cases $\frac{\Delta n^\mathrm{c}_i}{\Delta n^\mathrm{c}_{i-1}}$"
)
axs[2][1].set_ylabel(
    r"daily growth factor deaths $\frac{\Delta n^\mathrm{d}_i}{\Delta n^\mathrm{d}_{i-1}}$"
)

axs[3][0].set_ylabel("new cases per day")
axs[3][1].set_ylabel("new deaths per day")

axs[2][0].set_ylim(0, 5)
axs[2][1].set_ylim(0, 5)

for ax in axs[0]:
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 4))

for ax in axs[1]:
    ax.set_yscale("log")

for ax_ in axs:
    for ax in ax_:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=15, maxticks=52))
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlim(pd.Timestamp("2019-12-31"), pd.Timestamp.now())
        ax.legend(loc=2)
        ax.grid(True)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(
        f"linlog_world.{ext}", dpi=200,
    )
# -







plt.plot(
    *evf(
        np.linspace(-6, 6),
        lambda x, L, k, x0: 1 / (1 + np.exp(-k * (x - x0))),
        L=1,
        k=1,
        x0=0,
    )
)
plt.gca().set_yscale("log")





