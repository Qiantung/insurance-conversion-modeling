"""Run the original WOE binner on synthetic data; never read customer files."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from optimized_homework import WoeBinner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/synthetic'))
    args = parser.parse_args()
    rng = np.random.default_rng(20261110)
    count = 2400
    x = pd.DataFrame({
        'synthetic_engagement': rng.normal(size=count),
        'synthetic_tenure': rng.uniform(0, 5, size=count),
        'synthetic_channel': rng.choice(['A', 'B', 'C'], size=count),
    })
    logit = -1.3 + .9*x.synthetic_engagement + .2*x.synthetic_tenure
    y = rng.binomial(1, 1/(1+np.exp(-logit.to_numpy())))
    x.loc[rng.choice(count, 80, replace=False), 'synthetic_engagement'] = np.nan
    boundary = 1800
    binner = WoeBinner().fit(x.iloc[:boundary], y[:boundary])
    train = binner.transform(x.iloc[:boundary])
    test = binner.transform(x.iloc[boundary:])
    model = LogisticRegression(max_iter=1000).fit(train, y[:boundary])
    probability = model.predict_proba(test)[:, 1]
    metrics = {
        'data_type': 'SYNTHETIC', 'seed': 20261110,
        'train_rows': boundary, 'test_rows': count-boundary,
        'features': list(x.columns), 'bin_count': len(binner.rows),
        'test_auc': float(roc_auc_score(y[boundary:], probability)),
        'test_brier': float(brier_score_loss(y[boundary:], probability)),
        'scope': 'WOE binner + logistic regression demonstration; not original customer results',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame(binner.rows).to_csv(args.output_dir/'synthetic_bins.csv', index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
