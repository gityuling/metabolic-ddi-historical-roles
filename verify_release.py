"""Verify the published benchmark from a Zenodo ZIP using only CPU dependencies."""
import argparse
import json
import io
from zipfile import ZipFile

import numpy as np
import pandas as pd


def verify(archive):
    with ZipFile(archive) as data:
        targets = pd.read_csv(data.open('comparison.csv'))
        scores = []
        for row in targets.itertuples():
            test = pd.read_csv(data.open(f'{row.dataset}/test.csv'))
            y = test.target.to_numpy()
            values = np.load(io.BytesIO(data.read(f'{row.dataset}/{row.method}.npy')), allow_pickle=False)
            assert values.shape == (5, row.test_pairs, 4)
            np.testing.assert_allclose(values.sum(axis=2), 1, atol=2e-6)
            labels = np.r_[y, (y + 2) % 4]
            aligned = np.concatenate([values, values[:, :, [2, 3, 0, 1]]], axis=1)
            accuracy = (aligned.argmax(axis=2) == labels).mean(axis=1) * 100
            np.testing.assert_allclose(accuracy.mean(), row.accuracy, atol=1e-8, rtol=0)
            np.testing.assert_allclose(accuracy.std(ddof=1), row.accuracy_seed_sd, atol=1e-8, rtol=0)
            affected = np.where(y // 2 == 0, test.a_id, test.b_id)
            unique = (values == values.max(axis=2, keepdims=True)).sum(axis=2) == 1
            hits = (values.argmax(axis=2) == y) & unique
            paired = []
            for seed_hits in hits:
                trials = pd.DataFrame({'drug': affected, 'direction': y % 2, 'hit': seed_hits})
                by_drug = trials.groupby(['drug', 'direction']).hit.mean().unstack().dropna()
                paired.append(100 * (by_drug[0] * by_drug[1]).mean())
            paired_stats = json.loads(data.read(f'{row.dataset}/direction-summary.json'))
            key = 'MeTDDI checkpoint' if row.method == 'MeTDDI' else row.method
            np.testing.assert_allclose(paired, paired_stats['methods'][key]['per_seed'], atol=1e-8, rtol=0)
            scores.append((row.dataset, row.method, accuracy.mean()))
        return pd.DataFrame(scores, columns=['dataset', 'method', 'accuracy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--plot', help='Save a PDF or PNG comparison plot')
    args = parser.parse_args()
    result = verify(args.archive)
    print(result.to_string(index=False, float_format='{:.4f}'.format))
    if args.plot:
        import matplotlib.pyplot as plt
        axes = result.pivot(index='dataset', columns='method', values='accuracy').plot.bar(rot=0)
        axes.set(xlabel='', ylabel='Four-class accuracy (%)', ylim=(0, 100))
        axes.figure.tight_layout()
        axes.figure.savefig(args.plot)
