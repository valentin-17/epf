
# Electricity Price Forecasting on the German day-ahead market

This repository contains the code for the bachelors thesis "Forecasting electricity prices in the German day-ahead market with machine learning algorithms".

The project-structure largely follows the how-to found here: https://drivendata.github.io/cookiecutter-data-science/

---

**Table of Contents**

- [Initial Feature Selection](#initial-feature-selection)
- [Data Preprocessing](#data-preprocessing)
  - [Feature Analysis](#feature-analysis)

---

### Initial Feature Selection

*All timeseries were exported in UTC time format for consistency.*

|          Feature           | Market / Bidding Zone |    Interval    | Format |        Start (UTC)         |         End (UTC)          |                                                                                                                                                                      Source                                                                                                                                                                      |
|:--------------------------:|:---------------------:|:--------------:|:------:|:--------------------------:|:--------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|           Prices           |         DE_LU         |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |                  [DE_LU Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&year=2023&interval=year&legendItems=fy6&timezone=utc)<br>[DE_LU Prices 2024](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&year=2024&interval=year&legendItems=ey5&timezone=utc)                  |
|            Load            |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |                     [DE Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyi&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyi&year=2024&interval=year&timezone=utc&source=entsoe)                     |
|      Solar Generation      |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |         [DE Solar Generation 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyh&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Solar Generation 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyh&year=2024&interval=year&timezone=utc&source=entsoe)         |
| Wind Generation (Onshore)  |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |  [DE Wind Generation Onshore 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyg&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Wind Generation Onshore 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyg&year=2024&interval=year&timezone=utc&source=entsoe)  |
| Wind Generation (Offshore) |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | [DE Wind Generation Offshore 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyf&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Wind Generation Offshore 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyf&year=2024&interval=year&timezone=utc&source=entsoe) |
|            Load            |          CH           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |                     [CH Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=CH&legendItems=cy7&year=2023&interval=year&source=entsoe&timezone=utc)<br>[CH Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=CH&legendItems=cy7&year=2024&interval=year&source=entsoe&timezone=utc)                     |
|            Load            |          DK           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |                     [DK Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DK&legendItems=fy9&year=2023&interval=year&source=entsoe&timezone=utc)<br>[DK Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DK&legendItems=fy9&year=2024&interval=year&source=entsoe&timezone=utc)                     |
|            Load            |          FR           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 |                     [FR Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=FR&legendItems=jye&year=2023&interval=year&source=entsoe&timezone=utc)<br>[FR Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=FR&legendItems=jye&year=2024&interval=year&source=entsoe&timezone=utc)                     |

### Data Preprocessing

Explanatory data analysis is done in ```epf/notebooks/exploratory_analysis.ipynb```.

#### Feature Analysis

For feature analysis all timeseries were read into pandas DataFrames. Timeseries with a quarter hourly frequency were 
aggregated to the hour level using mean gruping. All Dataframes were then merged into a single DataFrame.

The pearson correlation coefficients for the feature "Prices" are as follows:

![Price correlation coefficients](/reports/figures/de_lu_price_correlations.png)


