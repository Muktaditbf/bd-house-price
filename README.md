# Bangladesh House Price Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A machine learning model that predicts the sale price of a property in Bangladesh (in taka) from its floor area, bedrooms, bathrooms, floor number, city and neighbourhood.

It is trained on about 2,800 unique listings from Dhaka, Chattogram, Narayanganj, Gazipur and Cumilla. On average its prediction is within **15%** of the real price.

![Predicted vs real price](images/predictions.png)

## Results

5-fold cross-validation on the de-duplicated data:

| Model | Avg. error (MAE) | Avg. % error (MAPE) | R² |
|---|---|---|---|
| Random Forest (baseline) | ৳1.79M | 16.2% | 0.800 |
| **Extra Trees + Gradient Boosting** | **৳1.71M** | **15.1%** | **0.821** |

On a single held-out test set (20% of the data), the final model scores R² 0.840 against 0.813 for the baseline and 0.680 for linear regression.

## What made the difference

- **Removing duplicate listings.** 934 of the 3,865 rows are exact copies. Left in, the same flat ends up in both the training and test sets, and the baseline looks better than it is (R² 0.888 instead of about 0.80). Every score above is measured after removing them.
- **Predicting log(price).** Prices run from about ৳2M to over ৳100M. On a log scale a 10% miss counts the same for a small flat and a large one.
- **Keeping missing values as missing.** Plots and shops have no bedrooms listed. Filling the gap with a typical value hid that; leaving it empty lets the trees learn it.
- **Averaging two models.** Extra Trees and Histogram Gradient Boosting make different mistakes, so their average beats each one alone.

Ideas that were tested and did **not** help: neighbourhood price-per-sq-ft encoding, and a property-type feature (flat / plot / shop) taken from the listing title.

## Run it yourself

```bash
git clone https://github.com/Muktaditbf/bd-house-price.git
cd bd-house-price
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put `house_price_bd.csv` in the `data/` folder (see [data/README.md](data/README.md)), then open `house_price.ipynb` and run all cells. The last cells save the trained model to `house_price_model.joblib` and show how to price a new flat:

```python
predict_price(bedrooms=3, bathrooms=3, floor_no=5, floor_area=1500, city="dhaka", area="Dhanmondi")
# ৳16,127,543
```

## Project structure

```
bd-house-price/
├── house_price.ipynb   # cleaning, training, evaluation, prediction
├── data/               # dataset goes here (not committed)
├── images/             # charts used in this README
├── requirements.txt
└── LICENSE
```

## Limitations

- Listing prices are asking prices, not final sale prices.
- Neighbourhoods with only a few listings get less accurate predictions.
- Properties far above the typical price range (over ৳50M) tend to be under-predicted.

## License

[MIT](LICENSE)
