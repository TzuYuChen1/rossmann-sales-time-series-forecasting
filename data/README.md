# Data

## Rossmann Store Sales

This project uses daily sales data for Store 262 from the Kaggle Rossmann Store Sales competition dataset.

| Attribute | Value |
|---|---|
| Store analyzed | Store 262 |
| Date range | 01/01/2013 – 07/31/2015 |
| Frequency | Daily (open days only, 6-day weekly cycle) |
| Observations | 942 (open days for Store 262) |
| Target variable | `Sales` (daily revenue, EUR) |

## Download

Source: [https://www.kaggle.com/competitions/rossmann-store-sales/overview](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

Download `train.csv` from the competition data page.

## Setup

Raw data is not included in this repository. After downloading, place `train.csv` in this `data/` folder. Both `sarima_analysis.Rmd` and `lstm_forecast.py` read from `data/train.csv` by default.
