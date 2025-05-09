
# Electricity Price Forecasting on the German day-ahead market

This repository contains the code for my bachelors thesis "Forecasting electricity prices in the German day-ahead market with machine learning algorithms". The project-structure largely follows the how-to from cookiecutter data science [[1]](./references/refs.md).

---

### Available raw data

*All timeseries were exported in UTC time format for consistency.*

|          Feature           | Market / Bidding Zone |   Frequency    | Format |        Start (UTC)         |         End (UTC)          | Source  |                                                                                                                                                                       URL                                                                                                                                                                        |
|:--------------------------:|:---------------------:|:--------------:|:------:|:--------------------------:|:--------------------------:|---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|           Prices           |         DE_LU         |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                  [DE_LU Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&year=2023&interval=year&legendItems=fy6&timezone=utc)<br>[DE_LU Prices 2024](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&year=2024&interval=year&legendItems=ey5&timezone=utc)                  |
|            Load            |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |                     [DE Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyi&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyi&year=2024&interval=year&timezone=utc&source=entsoe)                     |
|      Solar Generation      |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |         [DE Solar Generation 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyh&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Solar Generation 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyh&year=2024&interval=year&timezone=utc&source=entsoe)         |
| Wind Generation (Onshore)  |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |  [DE Wind Generation Onshore 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyg&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Wind Generation Onshore 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyg&year=2024&interval=year&timezone=utc&source=entsoe)  |
| Wind Generation (Offshore) |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E | [DE Wind Generation Offshore 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyf&year=2023&interval=year&timezone=utc&source=entsoe)<br>[DE Wind Generation Offshore 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&legendItems=nyf&year=2024&interval=year&timezone=utc&source=entsoe) |
|       Gas Generation       |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |           [DE Gas Generation 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny8&year=2023&interval=year&timezone=utc)<br>[DE Gas Generation 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny8&year=2024&interval=year&timezone=utc)           |
|    Hard Coal Generation    |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |     [DE Hard Coal Generation 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny6&year=2023&interval=year&timezone=utc)<br>[DE Hard Coal Generation 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny5&year=2024&interval=year&timezone=utc)     |
|     Lignite Generation     |          DE           | quarter-hourly |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:45+00:00 | ENTSO-E |       [DE Lignite Generation 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny5&year=2023&interval=year&timezone=utc)<br>[DE Lignite Generation 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=entsoe&legendItems=ny4&year=2024&interval=year&timezone=utc)       |
|            Load            |          CH           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                     [CH Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=CH&legendItems=cy7&year=2023&interval=year&source=entsoe&timezone=utc)<br>[CH Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=CH&legendItems=cy7&year=2024&interval=year&source=entsoe&timezone=utc)                     |
|            Load            |          DK           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                     [DK Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DK&legendItems=fy9&year=2023&interval=year&source=entsoe&timezone=utc)<br>[DK Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DK&legendItems=fy9&year=2024&interval=year&source=entsoe&timezone=utc)                     |
|            Load            |          FR           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                     [FR Load 2023](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=FR&legendItems=jye&year=2023&interval=year&source=entsoe&timezone=utc)<br>[FR Load 2024](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=FR&legendItems=jye&year=2024&interval=year&source=entsoe&timezone=utc)                     |
|           Prices           |          CH           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                     [CH Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=CH&legendItems=by4&year=2023&interval=year&timezone=utc)<br>[CH Prices 2024](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=CH&legendItems=by4&year=2024&interval=year&timezone=utc)                     |
|           Prices           |          DK1          |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                    [DK1 Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DK&legendItems=7y4&year=2023&interval=year&timezone=utc)<br>[DK1 Prices 2024](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DK&legendItems=7y4&year=2024&interval=year&timezone=utc)                    |
|           Prices           |          DK2          |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                    [DK2 Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DK&legendItems=7y5&year=2023&interval=year&timezone=utc)<br>[DK2 Prices 2024](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DK&legendItems=7y5&year=2024&interval=year&timezone=utc)                    |
|           Prices           |          FR           |     hourly     |  csv   | 2022-12-31<br>T23:00+00:00 | 2024-12-31<br>T22:00+00:00 | ENTSO-E |                    [FR Prices 2023](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=FR&legendItems=8y6&year=2023&interval=year&timezone=utc)<br>[FR Prices 2024 ](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=FR&legendItems=8y6&year=2024&interval=year&timezone=utc)                     |

### Feature set

The list of available features after outlier removal and seasonal decomposition is shown in the table below. 
The features are used to train the models for the price forecasting task. Based on the configuration file, different
features can be selected for training.

| Feature                                 | Outlier<br/> removal | Seasonal <br/> decomp. |
|-----------------------------------------|:--------------------:|:----------------------:|
| _Prices_                                |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;DE-LU           |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;CH              |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;DK1             |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;DK2             |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;FR              |          x           |           x            |
| _Price lags $^1$_                       |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;1-Hour          |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;12-Hour         |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;24-Hour         |          x           |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;168-Hour        |          x           |           x            |
| _German generation_                     |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;Solar           |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;Wind Onshore    |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;Wind Offshore   |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;Gas             |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;Lignite         |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;Hard Coal       |                      |           x            |
| _Load_                                  |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;DE              |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;CH              |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;DK              |                      |           x            |
| &nbsp;&nbsp;&nbsp;&nbsp;FR              |                      |           x            |
| _Dummies_                               |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;Month           |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;Day of the week |                      |                        |
| &nbsp;&nbsp;&nbsp;&nbsp;holiday         |                      |                        |

$^1$ only DE-LU bidding zone

### Models

There are a number of pretrained models available in the `models` folder. To access a model load it with pickle and 
extract all the information you need from the dictionary. The models are saved in the following format:

```python
{
  'model_name': str,
  'best_model': Keras.Sequential,
  'best_hps': kt.HyperParameters,
  'history': Keras.History,
  'train_mean': pd.DataFrame,
  'train_std': pd.DataFrame,
  'seasonal': dict,
  'window': WindowGenerator,
  'train_df': pd.DataFrame,
  'validation_df': pd.DataFrame,
  'test_df': pd.DataFrame,
}
```

Model naming convention is as follows: 
```python
{
  'mb': ['lstm', 'gru', 'conv'],
  'fs': 'features',
  'hl': 'hl0',
  'dr': ['drY', 'drN'],
}
```
with 
- mb = the model builder that has been used
- fs = the features that have been used
- hl = specifying the maximum hidden layers that are configured
- dr = whether to use dropout

Final name should look like this: lstm_all_features_hl5_drN. This would translate to a LSTM model with all features,
a maximum of 5 hidden layers and no dropout.

To train a new model please configure your settings in the `config.py` file.

The class EpfPipeline in `epf/pipeline.py` is the backbone of the epf library.
It exposes three methods: `train`, `evaluate` and `predict`. 
All of these methods can be called from a simple Jupyter notebook.

When passing a new `model_name` to the `train` method, the pipeline will automatically train a new model.
In that case the parameters prep_data has to be set to True and use_tuned_hyperparams has to be set to False. 
Otherwise, there is no associated training data and no tuned hyperparameters exist to train the new model.
