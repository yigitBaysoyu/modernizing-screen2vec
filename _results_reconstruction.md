```python
import pandas as pd
import matplotlib.pyplot as plt
import re, os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from os.path import basename, dirname
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
plt.rcdefaults()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

bigsize = (10, 7, ) # /2.54 = cm2inch
labelrot = 0
```

    C:\Users\Jonathan\Dev\screen2vec-revisited\.venv\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
_FIELD_RE = re.compile(r"^\s*([A-Za-z][^:]*?)\s*:\s*(.*?)\s*$")
_MISSING = {"", "-", "n/a", "na", "none", "null"}


def _to_bool(v):
    return v.strip().lower() in {"yes", "true", "1"}


def _to_seconds(v):
    return float(v.split()[0])


# report label -> (dict key, caster)
_FIELDS = {
    "Item Index":                          ("item_index", int),
    "JSON Path":                           ("json_path", str),
    "Embedding Path":                      ("embedding_path", str),
    "JSON Valid":                          ("json_valid", _to_bool),
    "Sequence Similarity (Gestalt)*":      ("sequence_similarity", float),
    "Levenshtein Distance (minified)":     ("levenshtein_distance", int),
    "Levenshtein Similarity (minified)*":  ("levenshtein_similarity", float),
    "App Match":                           ("app_match", _to_bool),
    "Activity Match":                      ("activity_match", _to_bool),
    "Component Count (GT)":                ("component_count_gt", int),
    "Component Count (Pred)":              ("component_count_pred", int),
    "Class Accuracy*":                     ("class_accuracy", float),
    "Average Bounds IoU*":                 ("average_bounds_iou", float),
    "Execution Time":                      ("execution_time", _to_seconds),
}


def parse_metrics_from_report_file(path):
    """Parse an evaluation report into a flat dict of metrics.

    The side-by-side JSON dump is skipped entirely; missing or unparsable
    values become None so every report yields the same set of keys.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # header (above the JSON table) + everything below "METRICS:"
    head, sep, tail = text.partition("GROUND TRUTH")
    relevant = head + tail.partition("METRICS:")[2] if sep else text

    out = {key: None for key, _ in _FIELDS.values()}
    for line in relevant.splitlines():
        match = _FIELD_RE.match(line)
        if match is None:
            continue
        label, raw = match.group(1), match.group(2)
        if label not in _FIELDS:
            continue
        key, cast = _FIELDS[label]
        if raw.lower() in _MISSING:
            continue
        try:
            out[key] = cast(raw)
        except (ValueError, IndexError):
            out[key] = None
    return out


def _process_report(g):
    # e.g. "reconstructions_4096_avg/0_report.txt" -> "4096_avg"
    # e.g. "reconstructions_4096_last/0_report.txt" -> "4096_last"
    # keeping the pooling suffix (avg/last) in the key prevents the
    # 4096_avg and 4096_last folders from clobbering each other.
    dimensions = dirname(g).removeprefix("reconstructions_")
    idx = basename(g).split("_")[0]
    return (dimensions, idx), parse_metrics_from_report_file(g)


# ------------------------------------------------------------------- entry ---

def parse_reconstruction_dataset(pattern="reconstructions*/*", max_workers=os.cpu_count()):
    """Glob reconstruction report files and parse them in parallel.

    Returns a dict keyed by (dimensions, idx) -> metrics dict.
    """
    paths = [g for g in glob(pattern) if "report" in g]

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, metrics in tqdm(pool.map(_process_report, paths, chunksize=8)):
            results[key] = metrics

    return results
```


```python
results = parse_reconstruction_dataset()
print("Parsing done.")

df = pd.DataFrame(results).T
df.index.names = ['dim', 'idx']
```

    1294135it [06:24, 3363.37it/s]
    

    Parsing done.
    

## Filters


```python
# keep only same indices among all samples
# idx_per_dim = df.index.to_frame(index=False).groupby('dim')['idx'].apply(set)
# common_idx = set.intersection(*idx_per_dim)
# df = df.loc[df.index.get_level_values('idx').isin(common_idx)]

# remove overtrained adapter
#df = df[(df.index.get_level_values('dim') != '128x4096_avg')]

# Take only validation set jsons into account
# df = df[(df.index.get_level_values('idx').astype(int) > 11504)]

# Take only valid jsons into account across all dimensions
# df = df[df.groupby(level='idx')['json_valid'].transform('all')]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>item_index</th>
      <th>json_path</th>
      <th>embedding_path</th>
      <th>json_valid</th>
      <th>sequence_similarity</th>
      <th>levenshtein_distance</th>
      <th>levenshtein_similarity</th>
      <th>app_match</th>
      <th>activity_match</th>
      <th>component_count_gt</th>
      <th>component_count_pred</th>
      <th>class_accuracy</th>
      <th>average_bounds_iou</th>
      <th>execution_time</th>
    </tr>
    <tr>
      <th>dim</th>
      <th>idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b</th>
      <th>0</th>
      <td>0</td>
      <td>../rico-data/data/layouts/0.json</td>
      <td>embeddings_128x4096_avg/0.npz</td>
      <td>True</td>
      <td>0.9107</td>
      <td>19</td>
      <td>0.8869</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>10000</td>
      <td>../rico-data/data/layouts/10000.json</td>
      <td>embeddings_128x4096_avg/10000.npz</td>
      <td>True</td>
      <td>0.3234</td>
      <td>1386</td>
      <td>0.3796</td>
      <td>False</td>
      <td>True</td>
      <td>9</td>
      <td>12</td>
      <td>0.6667</td>
      <td>0.1542</td>
      <td>13.23</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>10002</td>
      <td>../rico-data/data/layouts/10002.json</td>
      <td>embeddings_128x4096_avg/10002.npz</td>
      <td>False</td>
      <td>0.14</td>
      <td>1929</td>
      <td>0.5294</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.38</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>10003</td>
      <td>../rico-data/data/layouts/10003.json</td>
      <td>embeddings_128x4096_avg/10003.npz</td>
      <td>False</td>
      <td>0.091</td>
      <td>3068</td>
      <td>0.4347</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.76</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>10004</td>
      <td>../rico-data/data/layouts/10004.json</td>
      <td>embeddings_128x4096_avg/10004.npz</td>
      <td>True</td>
      <td>0.343</td>
      <td>619</td>
      <td>0.5156</td>
      <td>False</td>
      <td>True</td>
      <td>9</td>
      <td>14</td>
      <td>0.6667</td>
      <td>0.1267</td>
      <td>9.07</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b</th>
      <th>9998</th>
      <td>9998</td>
      <td>../rico-data/data/layouts/9998.json</td>
      <td>embeddings_8x4096_avg/9998.npz</td>
      <td>True</td>
      <td>0.8026</td>
      <td>616</td>
      <td>0.7528</td>
      <td>True</td>
      <td>True</td>
      <td>26</td>
      <td>20</td>
      <td>1.0</td>
      <td>0.9356</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>9999</td>
      <td>../rico-data/data/layouts/9999.json</td>
      <td>embeddings_8x4096_avg/9999.npz</td>
      <td>False</td>
      <td>0.0292</td>
      <td>10439</td>
      <td>0.209</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>../rico-data/data/layouts/999.json</td>
      <td>embeddings_8x4096_avg/999.npz</td>
      <td>False</td>
      <td>0.0438</td>
      <td>12970</td>
      <td>0.145</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>../rico-data/data/layouts/99.json</td>
      <td>embeddings_8x4096_avg/99.npz</td>
      <td>True</td>
      <td>0.631</td>
      <td>150</td>
      <td>0.789</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>0.6</td>
      <td>0.839</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>../rico-data/data/layouts/9.json</td>
      <td>embeddings_8x4096_avg/9.npz</td>
      <td>True</td>
      <td>0.2523</td>
      <td>441</td>
      <td>0.6152</td>
      <td>False</td>
      <td>False</td>
      <td>8</td>
      <td>7</td>
      <td>0.4286</td>
      <td>0.0154</td>
      <td>0.87</td>
    </tr>
  </tbody>
</table>
<p>1294135 rows × 14 columns</p>
</div>




```python
df.columns
```




    Index(['item_index', 'json_path', 'embedding_path', 'json_valid',
           'sequence_similarity', 'levenshtein_distance', 'levenshtein_similarity',
           'app_match', 'activity_match', 'component_count_gt',
           'component_count_pred', 'class_accuracy', 'average_bounds_iou',
           'execution_time'],
          dtype='str')




```python
df.groupby("dim").count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_index</th>
      <th>json_path</th>
      <th>embedding_path</th>
      <th>json_valid</th>
      <th>sequence_similarity</th>
      <th>levenshtein_distance</th>
      <th>levenshtein_similarity</th>
      <th>app_match</th>
      <th>activity_match</th>
      <th>component_count_gt</th>
      <th>component_count_pred</th>
      <th>class_accuracy</th>
      <th>average_bounds_iou</th>
      <th>execution_time</th>
    </tr>
    <tr>
      <th>dim</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b</th>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base</th>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base</th>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base</th>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-4b-base</th>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b</th>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base</th>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base</th>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base</th>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
    </tr>
    <tr>
      <th>8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import BarContainer


FIRST_NUMBER_PRIORITY = {
    4096: 0,
    4: 1,
    8: 2,
    128: 3,
}


def first_number_sort_key(value):
    text = str(value)
    match = re.search(r"\d+", text)

    if match is None:
        return 2, float("inf"), text.casefold()

    first_number = int(match.group())

    if first_number in FIRST_NUMBER_PRIORITY:
        return (
            0,
            FIRST_NUMBER_PRIORITY[first_number],
            text.casefold(),
        )

    return (
        1,
        first_number,
        text.casefold(),
    )


def plot_bool_grouped(
    df: pd.DataFrame,
    bool_cols: list,
    by: str,
    as_percentage: bool = True,
    xlabel: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
):
    data = (
        df.reset_index()
        if by in df.index.names and by not in df.columns
        else df.copy()
    )

    if as_percentage:
        grouped = data.groupby(by)[bool_cols].mean() * 100
        errors = data.groupby(by)[bool_cols].sem().fillna(0) * 100
        default_xlabel = "Share True (%)"
        value_format = "%.1f"
    else:
        grouped = data.groupby(by)[bool_cols].sum()
        errors = data.groupby(by)[bool_cols].std().fillna(0)
        default_xlabel = "Count True"
        value_format = "%.0f"

    sorted_labels = sorted(
        grouped.index,
        key=first_number_sort_key,
    )

    grouped = grouped.reindex(sorted_labels)
    errors = errors.reindex(sorted_labels)

    ax = grouped.plot(
        kind="barh",
        xerr=errors,
        capsize=4,
        figsize=bigsize,
        legend=show_legend,
    )

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                fmt=value_format,
                padding=3,
                fontsize=8,
            )

    ax.set_xlabel(
        xlabel if xlabel is not None else default_xlabel
    )
    ax.set_ylabel(by)
    ax.set_title(
        title
        if title is not None
        else f'True distribution grouped by "{by}"'
    )
    ax.tick_params(
        axis="y",
        labelrotation=labelrot,
    )

    ax.invert_yaxis()

    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    plt.tight_layout()
    plt.show()

def show_stats(column: str, by: str = "dim"):
    global df

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    # "dim" is an index level (df has a ("dim", "idx") MultiIndex), so it
    # cannot be addressed as df["dim"] -- fall back to the index level.
    labels = (
        df.index.get_level_values(by)
        if by in df.index.names and by not in df.columns
        else df[by]
    )
    label_values = pd.Series(labels, index=df.index)

    sorted_labels = sorted(
        label_values.dropna().unique(),
        key=first_number_sort_key,
    )

    boxplot_data = [
        df.loc[(label_values == label).to_numpy(), column].dropna()
        for label in sorted_labels
    ]

    fig, ax = plt.subplots(figsize=bigsize)

    ax.boxplot(
        boxplot_data,
        tick_labels=sorted_labels,
        showfliers=False,
        vert=False,
    )

    ax.set_xlabel(column)
    ax.set_ylabel(by)
    ax.set_title(f'{column} grouped by "{by}"')
    ax.tick_params(
        axis="y",
        labelrotation=labelrot,
    )

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    grouped_column = df.groupby(
        by,
        sort=False,
    )[column]

    desc_stats = grouped_column.describe()
    desc_stats["median"] = grouped_column.median()
    desc_stats["IQR"] = (
        grouped_column.quantile(0.75)
        - grouped_column.quantile(0.25)
    )

    desc_stats = desc_stats.reindex(sorted_labels)

    print(desc_stats)

```


```python
df.groupby('dim').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_index</th>
      <th>json_path</th>
      <th>embedding_path</th>
      <th>json_valid</th>
      <th>sequence_similarity</th>
      <th>levenshtein_distance</th>
      <th>levenshtein_similarity</th>
      <th>app_match</th>
      <th>activity_match</th>
      <th>component_count_gt</th>
      <th>component_count_pred</th>
      <th>class_accuracy</th>
      <th>average_bounds_iou</th>
      <th>execution_time</th>
    </tr>
    <tr>
      <th>dim</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b</th>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
      <td>46927</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base</th>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
      <td>3797</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base</th>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base</th>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
      <td>33733</td>
    </tr>
    <tr>
      <th>4096_last__via__embeddings_4096_last_full_qwen3-4b-base</th>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
      <td>725</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b</th>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base</th>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
      <td>66260</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base</th>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
      <td>2773</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
    <tr>
      <th>64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base</th>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
      <td>12757</td>
    </tr>
    <tr>
      <th>8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b</th>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
      <td>66261</td>
    </tr>
  </tbody>
</table>
</div>




```python
file_counts_df = (
    pd.Series(
        {
            directory.removeprefix("reconstructions_"):
                len(glob(f"{directory}/*")) // 2
            for directory in glob("reconstructions_*")
        },
        name="num_items",
    )
    .rename_axis("dim")
    .reset_index()
)


plot_bool_grouped(
    file_counts_df,
    ["num_items"],
    by="dim",
    as_percentage=False,
    xlabel="Num Items (Files / 2)",
    title='Num Items per "dim" (on filesystem, unfiltered)',
    show_legend=False,
)

plot_bool_grouped(
    df,
    ["json_valid"],
    by="dim",
    as_percentage=False,
)

plot_bool_grouped(
    df,
    ["app_match"],
    by="dim",
    as_percentage=False,
)

plot_bool_grouped(
    df,
    ["activity_match"],
    by="dim",
    as_percentage=False,
)
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    


## Boxplot: Sequence Similarity


```python
show_stats('sequence_similarity')
```


    
![png](output_12_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.220665   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.202299   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.212277   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.077611   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.223118   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.229340   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.231678   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.091386   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.268161   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.260890   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.270988   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.266362   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.255757   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.277734   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.600557   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.250717   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.241424   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.263688   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.336688   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.292741   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.279038   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.555821   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.283694   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.296952   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.868017   
    
                                                                        std  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.246057   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.212315   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.235985   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.071632   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.260560   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.254921   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.269281   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.077518   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.260755   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.255810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.259491   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.263941   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.254494   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.263674   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.359384   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.256303   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.246729   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.259987   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.267661   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.270432   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.270158   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.355750   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.268490   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.276273   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.239431   
    
                                                                      min     25%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0005  0.0485   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0004  0.0608   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0011  0.0478   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0007  0.0246   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.0420   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0007  0.0519   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0006  0.0426   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0005  0.0329   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0006  0.0746   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0003  0.0698   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0000  0.0725   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0005  0.0673   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0003  0.0680   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0002  0.0791   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0001  0.1774   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0007  0.0590   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.0632   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0006  0.0681   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0018  0.1327   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0004  0.0864   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0005  0.0709   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0001  0.1734   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0000  0.0818   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0005  0.0840   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.0000  0.8822   
    
                                                                      50%     75%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.1252  0.2945   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1311  0.2586   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.1233  0.2828   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0570  0.1072   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.1177  0.2933   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.1322  0.2960   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.1196  0.3068   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0668  0.1321   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.1769  0.3675   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.1728  0.3597   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.1843  0.3777   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.1718  0.3694   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.1647  0.3534   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.1856  0.3874   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.6822  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.1575  0.3486   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.1513  0.3273   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.1729  0.3633   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2514  0.4587   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2000  0.4057   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.1830  0.3947   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.6222  0.8843   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.1894  0.3926   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2011  0.4186   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.9722  1.0000   
    
                                                                      max  median  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.0000  0.1252   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.0000  0.1311   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.0000  0.1233   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.4151  0.0570   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.0000  0.1177   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.0000  0.1322   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.0000  0.1196   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.4206  0.0668   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.0000  0.1769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.0000  0.1728   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.0000  0.1843   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.0000  0.1718   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.0000  0.1647   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.0000  0.1856   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  0.6822   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.0000  0.1575   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.0000  0.1513   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.0000  0.1729   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.0000  0.2514   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.0000  0.2000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.0000  0.1830   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  0.6222   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.0000  0.1894   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.0000  0.2011   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.9722   
    
                                                                      IQR  
    dim                                                                    
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.2460  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1978  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.2350  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0826  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.2513  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.2441  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.2642  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0992  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2929  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.2899  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.3052  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.3021  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.2854  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.3083  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.8226  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.2896  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.2641  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.2952  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.3260  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.3193  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.3238  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.7109  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.3108  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.3346  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.1178  
    

## Boxplot: Levenshtein Distance


```python
show_stats('levenshtein_distance')
```


    
![png](output_14_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                           mean  \
    dim                                                                           
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b            5347.506361   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base         3535.711942   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base         5163.237455   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          11853.777931   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b          5718.243446   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base       4790.578244   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base       5530.469096   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        11888.813793   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b        3306.470857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new    3494.982705   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b    3690.192919   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base     3742.763375   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b          3967.714025   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base     3494.176381   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       3053.360620   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b          4122.126485   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b          3743.554172   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b        3636.174326   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b     1549.240054   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base   3346.762877   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base   3952.913388   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     2278.006321   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base   3319.436199   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base   3618.947284   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base      603.297876   
    
                                                                            std  \
    dim                                                                           
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b            6268.649989   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base         5153.100216   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base         6209.192383   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          28373.390795   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b          6351.175599   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base       6019.204308   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base       5670.711496   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        28516.077328   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b        5236.141683   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new    5319.249015   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b    5671.984559   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base     5471.916453   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b          5788.664967   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base     5472.734364   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      15128.671743   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b          5939.884468   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b          5510.931987   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b        5473.362974   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b     1973.414792   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base   5345.460027   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base   5790.794749   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    13288.755560   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base   5321.821304   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base   5459.658932   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base     7277.803346   
    
                                                                      min     25%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b              0.0   649.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base           0.0   633.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base           0.0   647.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          4140.0  7451.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b            0.0   726.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base         0.0   711.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base         0.0   669.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        3443.0  7620.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b          0.0   484.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new      0.0   501.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b      0.0   461.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base       0.0   470.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b            0.0   546.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base       0.0   464.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base         0.0     0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b            0.0   479.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b            0.0   531.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b          0.0   495.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b       0.0   381.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base     0.0   450.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base     0.0   457.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base       0.0    23.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base     0.0   455.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base     0.0   440.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base       0.0     0.0   
    
                                                                      50%  \
    dim                                                                     
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           2234.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1681.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        2028.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          9086.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         2510.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      2029.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      2319.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        9254.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1376.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1442.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1286.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1298.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1548.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1292.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base        70.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1328.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1459.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1362.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1041.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1323.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1416.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base      73.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1229.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1444.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base      16.0   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           11145.00   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base         4793.00   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        10786.00   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          11651.00   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         11523.00   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base       9898.00   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      11336.00   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        11643.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b        4379.25   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new    4992.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b    4697.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base     6838.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b          5434.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base     4323.25   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       5342.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b          7759.00   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b          4753.00   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b        4990.00   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b     1980.00   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base   4040.00   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base   6503.00   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base      280.00   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base   3969.00   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base   4853.00   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base       53.00   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           749606.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        751871.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        732905.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          749215.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         743409.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      751270.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      107037.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        751694.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       750954.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   752018.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   750576.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    736743.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         749107.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    734791.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      734778.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         751457.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         734440.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       748731.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b     82867.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  751906.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  750161.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    752247.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  733121.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  751424.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    741976.0   
    
                                                                   median  \
    dim                                                                     
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           2234.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1681.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        2028.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          9086.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         2510.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      2029.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      2319.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        9254.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1376.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1442.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1286.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1298.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1548.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1292.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base        70.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1328.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1459.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1362.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1041.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1323.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1416.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base      73.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1229.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1444.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base      16.0   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           10496.00  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base         4160.00  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        10139.00  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base           4200.00  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         10797.00  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base       9187.00  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      10667.00  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base         4023.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b        3895.25  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new    4491.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b    4236.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base     6368.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b          4888.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base     3859.25  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       5342.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b          7280.00  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b          4222.00  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b        4495.00  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b     1599.00  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base   3590.00  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base   6046.00  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base      257.00  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base   3514.00  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base   4413.00  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base       53.00  
    

## Boxplot: Levenshtein Similarity


```python
show_stats('levenshtein_similarity')
```


    
![png](output_16_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.413917   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.427867   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.411802   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.303935   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.399773   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.416106   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.407394   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.319245   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.495116   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.479759   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.489895   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.488432   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.468967   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.503300   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.646843   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.471061   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.470462   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.488462   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.577837   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.514561   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.495973   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.680846   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.509466   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.512298   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.928115   
    
                                                                        std  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.234545   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.200513   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.227083   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.115377   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.250143   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.239948   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.250743   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.154006   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.217904   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.224111   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.228593   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.228797   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.231778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.222899   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.267070   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.227637   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.213275   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.225761   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.193719   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.227838   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.238962   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.247411   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.224579   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.239163   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.134474   
    
                                                                      min     25%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0037  0.2185   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0007  0.3221   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0021  0.2233   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0042  0.2167   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0016  0.1882   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0015  0.2301   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0033  0.1994   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0009  0.1994   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0012  0.3672   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0005  0.3406   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0001  0.3582   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0041  0.3336   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0026  0.3203   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0022  0.3795   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0052  0.4536   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0012  0.3178   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.3522   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0010  0.3474   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0093  0.4590   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0006  0.3799   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0028  0.3340   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0001  0.4979   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0000  0.3783   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0013  0.3537   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.0000  0.9237   
    
                                                                      50%     75%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.4152  0.5401   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.4338  0.5133   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.4207  0.5339   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.3224  0.3837   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3869  0.5324   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.4122  0.5295   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3958  0.5362   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.3148  0.4392   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.4877  0.5966   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.4809  0.5900   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.4925  0.6053   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.4857  0.6028   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.4738  0.5879   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.5032  0.6149   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.6057  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.4756  0.5851   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.4690  0.5668   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.4873  0.6020   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.5569  0.6742   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.5065  0.6318   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.4986  0.6279   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.6673  0.9269   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.4984  0.6200   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.5129  0.6495   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.9729  1.0000   
    
                                                                      max  median  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.0000  0.4152   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.0000  0.4338   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.0000  0.4207   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.7410  0.3224   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.0000  0.3869   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.0000  0.4122   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.0000  0.3958   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.7692  0.3148   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.0000  0.4877   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.0000  0.4809   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.0000  0.4925   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.0000  0.4857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.0000  0.4738   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.0000  0.5032   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  0.6057   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.0000  0.4756   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.0000  0.4690   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.0000  0.4873   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.0000  0.5569   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.0000  0.5065   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.0000  0.4986   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  0.6673   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.0000  0.4984   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.0000  0.5129   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.9729   
    
                                                                      IQR  
    dim                                                                    
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.3216  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1912  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.3106  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.1670  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3442  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.2994  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3368  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.2398  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2294  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.2494  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.2471  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.2692  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.2676  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2354  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.5464  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.2673  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.2146  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.2546  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2152  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2519  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2939  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.4290  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2417  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2958  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.0763  
    

## Boxplot: Class Accuracy


```python
show_stats('class_accuracy')
```


    
![png](output_18_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.214169   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.176336   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.197940   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.075162   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.211590   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.225525   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.218385   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.112200   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.299063   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.284036   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.280449   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.289396   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.277462   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.301069   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.358193   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.246897   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.249570   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.286904   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.347687   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.331853   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.312616   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.495711   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.316945   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.367260   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.864024   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.310913  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.269147  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.295365  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.125134  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.319237  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.315550  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.327156  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.184940  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.325967  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.322802  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.330174  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.334445  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.325848  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.330129  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.467874  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.317426  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.310134  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.332798  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.374824  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.341487  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.343292  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.456660  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.336960  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.356803  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.292326  0.0   
    
                                                                      25%     50%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0000  0.0408   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0000  0.0526   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0000  0.2000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0000  0.1765   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0000  0.1538   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0000  0.1667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0000  0.1579   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0000  0.2000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0000  0.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0000  0.1111   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.1250   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0000  0.1667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0000  0.2273   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0000  0.2400   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0000  0.2000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0000  0.4231   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0000  0.2121   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0000  0.2857   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.9333  1.0000   
    
                                                                      75%     max  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.3333  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.2500  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.3056  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.1071  0.6173   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3333  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.3500  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3333  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.1944  0.8750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.4583  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.4615  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.4615  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.3913  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.4000  1.0000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.5000  1.0000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.6667  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.5455  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.5000  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.5000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.6250  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    
                                                                   median     IQR  
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.3333  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0408  0.2500  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.3056  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.1071  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.3333  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0526  0.3500  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.3333  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.1944  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2000  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.1765  0.4583  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.1538  0.4615  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.1667  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.1579  0.4615  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2000  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0000  1.0000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.1111  0.3913  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.1250  0.4000  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.1667  0.5000  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2273  0.6667  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2400  0.5455  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2000  0.5000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.4231  1.0000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2121  0.5000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2857  0.6250  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.0667  
    

## Boxplot: Average Bounds IoU


```python
show_stats('average_bounds_iou')
```


    
![png](output_20_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.147447   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.139067   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.139051   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.026045   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.154922   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.158880   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.165071   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.039916   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.191700   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.181409   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.189034   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.188953   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.175004   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.193142   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.594613   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.167591   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.161541   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.184711   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.211662   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.211403   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.203607   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.538733   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.203827   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.217131   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.836202   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.275435  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.250852  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.263060  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.041466  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.286566  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.283856  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.298051  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.079156  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.288102  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.284026  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.289884  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.294379  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.282554  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.290274  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.463120  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.281591  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.274752  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.289390  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.314504  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.301569  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.301400  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.443979  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.296571  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.307106  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.294575  0.0   
    
                                                                      25%     50%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.0180   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0030  0.0410   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.0199   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.0094   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.0133   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0000  0.0305   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.0180   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.0001   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0084  0.0710   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0037  0.0625   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0000  0.0652   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0000  0.0604   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0000  0.0542   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0023  0.0708   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0071  0.9175   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0000  0.0450   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.0441   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0000  0.0629   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0000  0.0642   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0047  0.0833   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0000  0.0718   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0248  0.6013   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0043  0.0792   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0000  0.0867   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.8080  1.0000   
    
                                                                      75%     max  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.1367  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1289  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.1308  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0326  0.3926   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.1454  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.1508  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.1522  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0481  0.7958   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2043  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.1892  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.2067  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.2041  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.1823  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2105  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.1658  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.1572  1.0000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.1958  1.0000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2648  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2443  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2341  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2265  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2552  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    
                                                                   median     IQR  
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0180  0.1367  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0410  0.1259  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0199  0.1308  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0094  0.0326  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0133  0.1454  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0305  0.1508  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0180  0.1522  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0001  0.0481  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0710  0.1959  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0625  0.1855  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0652  0.2067  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0604  0.2041  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0542  0.1823  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0708  0.2082  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.9175  0.9929  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0450  0.1658  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0441  0.1572  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0629  0.1958  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0642  0.2648  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0833  0.2396  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0718  0.2341  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.6013  0.9752  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0792  0.2222  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0867  0.2552  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.1920  
    

## Repaired JSON


```python
# --- Repair pass ---------------------------------------------------------------
#
# The reports on disk only hold the *strict* evaluation: a prediction that
# does not parse counts as 0 components, 0 class accuracy, 0 IoU. Here the raw
# predictions (`*_reconstructed.json`) are loaded again, repaired with
# json_repair, and scored with exactly the same functions that produced the
# reports: reconstruction_metrics.score_against_ground_truth(), shared with
# 3_json_reconstruction_par.py.
#
# The strict columns are left untouched; everything repaired is added
# alongside under the "_repaired" suffix -- otherwise the numbers here would
# no longer be comparable to the ones in the reports.

from concurrent.futures import ProcessPoolExecutor

import repair_rescore

# ---------------------------------------------------------------- Switches ---
# Collapse repetition loops in the LLM output: a degenerate generation does
# not stop after the screen, it repeats the same component (or the same short
# cycle) until the token budget runs out. Every copy shifts the paired
# comparison with the ground truth by one position, so exact-duplicate
# sibling subtrees are collapsed onto a single one, recursively.
COLLAPSE_LOOPS = True

# Only rescore the broken JSONs. For a prediction that already parsed, the
# repair is a no-op -- the strict value *is* the repaired one there. But: a
# valid JSON can be looped too (roughly one in five rows), and this pass then
# never sees it. So True = fast (only the invalid rows), False = complete, at
# roughly 190 rows/s on 14 cores.
ONLY_INVALID = False

MAX_WORKERS = None   # None -> os.cpu_count()
CHUNKSIZE = 16

tasks = repair_rescore.build_tasks(df, only_invalid=ONLY_INVALID)
n_rows = sum(len(entries) for _, _, entries in tasks)

print(
    f"Rescoring {n_rows} of {len(df)} rows "
    f"({len(tasks)} screens, COLLAPSE_LOOPS={COLLAPSE_LOOPS})"
)
```

    Rescoring 1294135 of 1294135 rows (66261 screens, COLLAPSE_LOOPS=True)
    


```python
# Processes, not threads: cleaning the Rico ground truth, parsing, and the
# Levenshtein distance are CPU-bound and would otherwise block each other on
# the GIL.
#
# One task is a screen index together with every dimension that needs it --
# the ground truth is loaded once per screen instead of once per row.

rows = []

if tasks:
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=repair_rescore.init_worker,
        initargs=(COLLAPSE_LOOPS,),
    ) as pool:
        for result in tqdm(
            pool.map(repair_rescore.rescore_item, tasks, chunksize=CHUNKSIZE),
            total=len(tasks),
            desc="repair + rescore",
        ):
            rows.extend(result)

df_repaired = (
    pd.DataFrame(rows).set_index(["dim", "idx"])
    if rows
    else pd.DataFrame(index=df.index[:0])
)

if "rescore_error" in df_repaired.columns:
    n_failed = int(df_repaired["rescore_error"].notna().sum())
    if n_failed:
        print(f"{n_failed} rows without a result (prediction file missing)")

df_repaired.head()
```

    repair + rescore: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 66261/66261 [35:27<00:00, 31.14it/s]
    

    1 rows without a result (prediction file missing)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>json_valid_repaired</th>
      <th>json_repair_used</th>
      <th>components_collapsed</th>
      <th>levenshtein_distance_repaired</th>
      <th>levenshtein_similarity_repaired</th>
      <th>app_match_repaired</th>
      <th>activity_match_repaired</th>
      <th>component_count_gt_repaired</th>
      <th>component_count_pred_repaired</th>
      <th>class_accuracy_repaired</th>
      <th>average_bounds_iou_repaired</th>
      <th>class_unordered_recall</th>
      <th>class_unordered_recall_repaired</th>
      <th>class_unordered_precision</th>
      <th>class_unordered_precision_repaired</th>
      <th>class_unordered_f1</th>
      <th>class_unordered_f1_repaired</th>
      <th>class_level_recall</th>
      <th>class_level_recall_repaired</th>
      <th>class_level_precision</th>
      <th>class_level_precision_repaired</th>
      <th>class_level_f1</th>
      <th>class_level_f1_repaired</th>
      <th>matched_iou</th>
      <th>matched_iou_repaired</th>
      <th>rescore_error</th>
    </tr>
    <tr>
      <th>dim</th>
      <th>idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b</th>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.886905</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base</th>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.886905</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base</th>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>0.904762</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base</th>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>121.0</td>
      <td>0.447489</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.929293</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b</th>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.887574</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Carry the repaired columns for *all* rows, not just the rescored ones:
# where a row was not rescored, the strict value already is the repaired one.
# Otherwise every plot below would be comparing two different populations.

REPAIRED_TWIN = {
    "json_valid_repaired":             "json_valid",
    "levenshtein_distance_repaired":   "levenshtein_distance",
    "levenshtein_similarity_repaired": "levenshtein_similarity",
    "app_match_repaired":              "app_match",
    "activity_match_repaired":         "activity_match",
    "component_count_gt_repaired":     "component_count_gt",
    "component_count_pred_repaired":   "component_count_pred",
    "class_accuracy_repaired":         "class_accuracy",
    "average_bounds_iou_repaired":     "average_bounds_iou",
}

# The order-free metrics (class_unordered_*, class_level_*, matched_iou) were
# added after the reports on disk were written, so -- unlike every column above
# -- they have no strict twin in `df` to fall back on. Both halves come out of
# the rescore pass, which is why they are carried separately below and why a
# row that was not rescored stays NaN rather than silently borrowing a value.
from reconstruction_metrics import UNORDERED_METRIC_KEYS

_empty = pd.Series(index=df.index, dtype="object")

for repaired_col, strict_col in REPAIRED_TWIN.items():
    values = (
        df_repaired[repaired_col].reindex(df.index)
        if repaired_col in df_repaired.columns
        else _empty
    )
    df[repaired_col] = values.where(values.notna(), df[strict_col])

for key in UNORDERED_METRIC_KEYS:
    for col in (key, f"{key}_repaired"):
        df[col] = pd.to_numeric(
            df_repaired[col].reindex(df.index)
            if col in df_repaired.columns
            else _empty,
            errors="coerce",
        )

n_unscored = int(df["class_unordered_f1"].isna().sum())
if n_unscored:
    print(
        f"{n_unscored} of {len(df)} rows carry no order-free score -- they were "
        "not part of the rescore pass. Set ONLY_INVALID = False above for full "
        "coverage."
    )


# Only ever defined for the rescored rows; everywhere else is "not repaired"
# resp. "nothing collapsed" by construction.
for extra_col, default in (("json_repair_used", False), ("components_collapsed", 0)):
    values = (
        df_repaired[extra_col].reindex(df.index)
        if extra_col in df_repaired.columns
        else _empty
    )
    df[extra_col] = values.where(values.notna(), default)


# ---------------------------------------------------------- Summary table ---
summary_cols = [
    "json_valid", "json_valid_repaired", "json_repair_used",
    "app_match", "app_match_repaired",
    "activity_match", "activity_match_repaired",
    "class_accuracy", "class_accuracy_repaired",
    "average_bounds_iou", "average_bounds_iou_repaired",
    "levenshtein_similarity", "levenshtein_similarity_repaired",
    "component_count_gt", "component_count_pred", "component_count_pred_repaired",
    "components_collapsed",
] + [
    col
    for key in UNORDERED_METRIC_KEYS
    for col in (key, f"{key}_repaired")
]

summary = (
    df[summary_cols]
    .apply(pd.to_numeric, errors="coerce")
    .astype(float)
    .groupby(df.index.get_level_values("dim"))
    .mean()
    .rename_axis("dim")
)
summary.insert(0, "n", df.groupby("dim").size())
summary = summary.reindex(sorted(summary.index, key=first_number_sort_key))

print(summary.T.to_string(float_format=lambda v: f"{v:.4f}"))
```

    1 of 1294135 rows carry no order-free score -- they were not part of the rescore pass. Set ONLY_INVALID = False above for full coverage.
    dim                                 4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b  4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base  4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b  4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base  4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base  4096_last__via__embeddings_4096_last_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b  8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b  128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base
    n                                                                             66261.0000                                               66261.0000                                               66261.0000                                               725.0000                                              66261.0000                                                 66261.0000                                                 33733.0000                                                 725.0000                                                66261.0000                                                    66261.0000                                                    66261.0000                                                   66261.0000                                              66261.0000                                                   66261.0000                                                  2773.0000                                              66261.0000                                              66261.0000                                                66261.0000                                                   46927.0000                                                     66261.0000                                                     66261.0000                                                    3797.0000                                                     66261.0000                                                     66261.0000                                                   12757.0000
    json_valid                                                                        0.6136                                                   0.8208                                                   0.6378                                                 0.6510                                                  0.5895                                                     0.6919                                                     0.6136                                                   0.5145                                                    0.8218                                                        0.8021                                                        0.7620                                                       0.7520                                                  0.7438                                                       0.7782                                                     0.8457                                                  0.7435                                                  0.7756                                                    0.7374                                                       0.6300                                                         0.7858                                                         0.7395                                                       0.8918                                                         0.7841                                                         0.7544                                                       0.9614
    json_valid_repaired                                                               0.9729                                                   0.9989                                                   0.9882                                                 0.9972                                                  0.9597                                                     0.9921                                                     0.9845                                                   0.9931                                                    0.9735                                                        0.9797                                                        0.9930                                                       0.9899                                                  0.9845                                                       0.9847                                                     0.9957                                                  0.9869                                                  0.9917                                                    0.9740                                                       0.9983                                                         0.9854                                                         0.9878                                                       0.9934                                                         0.9872                                                         0.9908                                                       0.9944
    json_repair_used                                                                  0.3594                                                   0.1780                                                   0.3504                                                 0.3462                                                  0.3702                                                     0.3003                                                     0.3709                                                   0.4786                                                    0.1517                                                        0.1776                                                        0.2309                                                       0.2380                                                  0.2407                                                       0.2065                                                     0.1500                                                  0.2434                                                  0.2161                                                    0.2366                                                       0.3683                                                         0.1996                                                         0.2483                                                       0.1017                                                         0.2031                                                         0.2364                                                       0.0329
    app_match                                                                         0.0646                                                   0.0548                                                   0.0589                                                 0.0000                                                  0.0750                                                     0.0702                                                     0.0743                                                   0.0110                                                    0.0681                                                        0.0659                                                        0.0687                                                       0.0711                                                  0.0652                                                       0.0682                                                     0.2618                                                  0.0702                                                  0.0650                                                    0.0630                                                       0.0766                                                         0.0721                                                         0.0677                                                       0.1862                                                         0.0722                                                         0.0723                                                       0.5183
    app_match_repaired                                                                0.0715                                                   0.0593                                                   0.0647                                                 0.0000                                                  0.0870                                                     0.0760                                                     0.0824                                                   0.0166                                                    0.0741                                                        0.0715                                                        0.0768                                                       0.0737                                                  0.0702                                                       0.0751                                                     0.2618                                                  0.0753                                                  0.0697                                                    0.0730                                                       0.0889                                                         0.0765                                                         0.0760                                                       0.1862                                                         0.0749                                                         0.0844                                                       0.5207
    activity_match                                                                    0.1903                                                   0.1992                                                   0.1834                                                 0.1793                                                  0.2202                                                     0.2242                                                     0.2215                                                   0.1986                                                    0.2522                                                        0.2403                                                        0.2395                                                       0.2351                                                  0.2327                                                       0.2493                                                     0.3675                                                  0.1985                                                  0.2075                                                    0.2266                                                       0.2446                                                         0.2641                                                         0.2437                                                       0.4016                                                         0.2829                                                         0.2551                                                       0.7180
    activity_match_repaired                                                           0.2729                                                   0.2365                                                   0.2628                                                 0.2152                                                  0.3092                                                     0.2889                                                     0.3097                                                   0.3186                                                    0.2899                                                        0.2830                                                        0.2942                                                       0.2940                                                  0.2883                                                       0.3023                                                     0.3992                                                  0.2443                                                  0.2459                                                    0.2866                                                       0.3635                                                         0.3145                                                         0.3039                                                       0.4206                                                         0.3398                                                         0.3126                                                       0.7298
    class_accuracy                                                                    0.2142                                                   0.1763                                                   0.1979                                                 0.0752                                                  0.2116                                                     0.2255                                                     0.2184                                                   0.1122                                                    0.2991                                                        0.2840                                                        0.2804                                                       0.2894                                                  0.2775                                                       0.3011                                                     0.3582                                                  0.2469                                                  0.2496                                                    0.2869                                                       0.3477                                                         0.3319                                                         0.3126                                                       0.4957                                                         0.3169                                                         0.3673                                                       0.8640
    class_accuracy_repaired                                                           0.2935                                                   0.2120                                                   0.2718                                                 0.1011                                                  0.2788                                                     0.2848                                                     0.2794                                                   0.2039                                                    0.3332                                                        0.3228                                                        0.3383                                                       0.3411                                                  0.3353                                                       0.3520                                                     0.3859                                                  0.2967                                                  0.2939                                                    0.3415                                                       0.4642                                                         0.3857                                                         0.3808                                                       0.5160                                                         0.3705                                                         0.4495                                                       0.8741
    average_bounds_iou                                                                0.1474                                                   0.1391                                                   0.1391                                                 0.0260                                                  0.1549                                                     0.1589                                                     0.1651                                                   0.0399                                                    0.1917                                                        0.1814                                                        0.1890                                                       0.1890                                                  0.1750                                                       0.1931                                                     0.5946                                                  0.1676                                                  0.1615                                                    0.1847                                                       0.2117                                                         0.2114                                                         0.2036                                                       0.5387                                                         0.2038                                                         0.2171                                                       0.8362
    average_bounds_iou_repaired                                                       0.1953                                                   0.1609                                                   0.1790                                                 0.0494                                                  0.2014                                                     0.1981                                                     0.2131                                                   0.0764                                                    0.2252                                                        0.2173                                                        0.2291                                                       0.2251                                                  0.2096                                                       0.2265                                                     0.6094                                                  0.2006                                                  0.1868                                                    0.2255                                                       0.2674                                                         0.2470                                                         0.2478                                                       0.5484                                                         0.2377                                                         0.2610                                                       0.8406
    levenshtein_similarity                                                            0.4139                                                   0.4279                                                   0.4118                                                 0.3039                                                  0.3998                                                     0.4161                                                     0.4074                                                   0.3192                                                    0.4951                                                        0.4798                                                        0.4899                                                       0.4884                                                  0.4690                                                       0.5033                                                     0.6468                                                  0.4711                                                  0.4705                                                    0.4885                                                       0.5778                                                         0.5146                                                         0.4960                                                       0.6808                                                         0.5095                                                         0.5123                                                       0.9281
    levenshtein_similarity_repaired                                                   0.4212                                                   0.4247                                                   0.4154                                                 0.2866                                                  0.4088                                                     0.4223                                                     0.4185                                                   0.2561                                                    0.4864                                                        0.4700                                                        0.4873                                                       0.4846                                                  0.4698                                                       0.4981                                                     0.6288                                                  0.4718                                                  0.4754                                                    0.4846                                                       0.5622                                                         0.5127                                                         0.4999                                                       0.6685                                                         0.5065                                                         0.5207                                                       0.9268
    component_count_gt                                                               10.1951                                                  16.7814                                                  10.9517                                                72.0317                                                 10.4940                                                    13.2913                                                    10.8561                                                  49.1669                                                   16.2344                                                       15.8904                                                       14.6295                                                      13.1308                                                 14.3449                                                      14.5472                                                    11.5096                                                 13.7575                                                 14.8791                                                   12.9133                                                       7.9510                                                        14.3586                                                        13.3624                                                       9.3392                                                        14.1136                                                        13.6778                                                       7.1333
    component_count_pred                                                             12.6139                                                  22.1314                                                  13.3747                                                31.1559                                                 10.9400                                                    14.6642                                                    11.7562                                                  19.3972                                                   19.9920                                                       19.4769                                                       14.0230                                                      16.2163                                                 15.7497                                                      16.5440                                                     7.0757                                                 16.1004                                                 17.6529                                                   14.8316                                                       8.1605                                                        17.0014                                                        15.7618                                                       7.0166                                                        15.9189                                                        17.2657                                                       6.8450
    component_count_pred_repaired                                                    33.6896                                                  35.0150                                                  36.7777                                                59.0745                                                 35.6959                                                    33.4273                                                    32.0522                                                  42.2028                                                   25.6109                                                       24.0598                                                       22.0955                                                      27.8067                                                 25.2355                                                      25.4315                                                    17.4309                                                 26.2560                                                 30.0116                                                   26.6333                                                      16.7202                                                        25.8613                                                        26.0869                                                      11.1807                                                        24.9828                                                        27.0079                                                       8.8734
    components_collapsed                                                             24.5060                                                   5.0252                                                  19.0884                                                10.3641                                                 20.3098                                                    17.5169                                                    26.1502                                                  28.8055                                                    9.7382                                                       13.7982                                                       18.8704                                                      17.2269                                                 14.9750                                                      12.7429                                                     7.8013                                                 19.1128                                                 12.5805                                                   13.0003                                                       1.4364                                                        10.8031                                                        15.6189                                                       6.1246                                                        10.8451                                                        12.7053                                                       0.4230
    class_unordered_recall                                                            0.3031                                                   0.2682                                                   0.2948                                                 0.1063                                                  0.2817                                                     0.3014                                                     0.2870                                                   0.1129                                                    0.4122                                                        0.3936                                                        0.3959                                                       0.3940                                                  0.3831                                                       0.4171                                                     0.3682                                                  0.3716                                                  0.3819                                                    0.3901                                                       0.4164                                                         0.4425                                                         0.4188                                                       0.5338                                                         0.4014                                                         0.4726                                                       0.8690
    class_unordered_recall_repaired                                                   0.3738                                                   0.2988                                                   0.3658                                                 0.1329                                                  0.3443                                                     0.3536                                                     0.3434                                                   0.1919                                                    0.4304                                                        0.4117                                                        0.4363                                                       0.4324                                                  0.4285                                                       0.4543                                                     0.3891                                                  0.4115                                                  0.4248                                                    0.4339                                                       0.5291                                                         0.4809                                                         0.4700                                                       0.5513                                                         0.4371                                                         0.5343                                                       0.8805
    class_unordered_precision                                                         0.2703                                                   0.2511                                                   0.2758                                                 0.2164                                                  0.2759                                                     0.3001                                                     0.2839                                                   0.2420                                                    0.3965                                                        0.3905                                                        0.4116                                                       0.3759                                                  0.3705                                                       0.4003                                                     0.3796                                                  0.3619                                                  0.3498                                                    0.3679                                                       0.4070                                                         0.4120                                                         0.3979                                                       0.5361                                                         0.3859                                                         0.4222                                                       0.8705
    class_unordered_precision_repaired                                                0.4099                                                   0.3066                                                   0.3920                                                 0.2447                                                  0.3768                                                     0.3908                                                     0.3818                                                   0.4342                                                    0.4752                                                        0.4884                                                        0.5261                                                       0.4741                                                  0.4751                                                       0.4992                                                     0.4347                                                  0.4648                                                  0.4255                                                    0.4651                                                       0.6042                                                         0.5008                                                         0.5102                                                       0.5788                                                         0.4718                                                         0.5344                                                       0.8893
    class_unordered_f1                                                                0.2791                                                   0.2516                                                   0.2781                                                 0.1388                                                  0.2693                                                     0.2884                                                     0.2756                                                   0.1380                                                    0.3939                                                        0.3805                                                        0.3966                                                       0.3766                                                  0.3672                                                       0.4015                                                     0.3702                                                  0.3604                                                  0.3591                                                    0.3723                                                       0.4078                                                         0.4186                                                         0.4002                                                       0.5303                                                         0.3874                                                         0.4371                                                       0.8696
    class_unordered_f1_repaired                                                       0.3479                                                   0.2781                                                   0.3390                                                 0.1571                                                  0.3220                                                     0.3338                                                     0.3255                                                   0.2257                                                    0.4247                                                        0.4140                                                        0.4445                                                       0.4208                                                  0.4180                                                       0.4473                                                     0.3937                                                  0.4059                                                  0.4020                                                    0.4224                                                       0.5415                                                         0.4638                                                         0.4588                                                       0.5516                                                         0.4298                                                         0.5038                                                       0.8828
    class_level_recall                                                                0.2414                                                   0.1887                                                   0.2247                                                 0.0557                                                  0.2339                                                     0.2380                                                     0.2396                                                   0.0662                                                    0.3325                                                        0.3168                                                        0.3282                                                       0.3222                                                  0.3116                                                       0.3459                                                     0.3558                                                  0.2984                                                  0.2969                                                    0.3249                                                       0.3806                                                         0.3794                                                         0.3588                                                       0.5151                                                         0.3434                                                         0.4155                                                       0.8637
    class_level_recall_repaired                                                       0.2783                                                   0.2029                                                   0.2589                                                 0.0631                                                  0.2693                                                     0.2645                                                     0.2710                                                   0.0978                                                    0.3454                                                        0.3302                                                        0.3538                                                       0.3416                                                  0.3396                                                       0.3703                                                     0.3636                                                  0.3197                                                  0.3184                                                    0.3527                                                       0.4608                                                         0.4041                                                         0.3923                                                       0.5241                                                         0.3653                                                         0.4592                                                       0.8711
    class_level_precision                                                             0.2173                                                   0.1786                                                   0.2106                                                 0.1164                                                  0.2273                                                     0.2334                                                     0.2349                                                   0.1399                                                    0.3221                                                        0.3142                                                        0.3375                                                       0.3069                                                  0.2980                                                       0.3331                                                     0.3599                                                  0.2920                                                  0.2749                                                    0.3078                                                       0.3709                                                         0.3533                                                         0.3414                                                       0.5128                                                         0.3298                                                         0.3728                                                       0.8647
    class_level_precision_repaired                                                    0.3027                                                   0.2146                                                   0.2774                                                 0.1258                                                  0.2898                                                     0.2865                                                     0.2933                                                   0.2314                                                    0.3793                                                        0.3839                                                        0.4165                                                       0.3693                                                  0.3676                                                       0.4037                                                     0.3866                                                  0.3543                                                  0.3195                                                    0.3774                                                       0.5139                                                         0.4195                                                         0.4236                                                       0.5373                                                         0.3912                                                         0.4597                                                       0.8780
    class_level_f1                                                                    0.2241                                                   0.1790                                                   0.2129                                                 0.0736                                                  0.2238                                                     0.2275                                                     0.2302                                                   0.0804                                                    0.3205                                                        0.3080                                                        0.3278                                                       0.3090                                                  0.2981                                                       0.3344                                                     0.3557                                                  0.2910                                                  0.2815                                                    0.3116                                                       0.3724                                                         0.3598                                                         0.3441                                                       0.5107                                                         0.3321                                                         0.3858                                                       0.8640
    class_level_f1_repaired                                                           0.2634                                                   0.1940                                                   0.2451                                                 0.0781                                                  0.2551                                                     0.2525                                                     0.2585                                                   0.1159                                                    0.3426                                                        0.3323                                                        0.3599                                                       0.3341                                                  0.3310                                                       0.3659                                                     0.3650                                                  0.3176                                                  0.3053                                                    0.3453                                                       0.4677                                                         0.3910                                                         0.3848                                                       0.5218                                                         0.3601                                                         0.4354                                                       0.8729
    matched_iou                                                                       0.1356                                                   0.1105                                                   0.1282                                                 0.0228                                                  0.1418                                                     0.1455                                                     0.1507                                                   0.0309                                                    0.1841                                                        0.1753                                                        0.1883                                                       0.1853                                                  0.1745                                                       0.1908                                                     0.3394                                                  0.1658                                                  0.1585                                                    0.1807                                                       0.2052                                                         0.2098                                                         0.2013                                                       0.4174                                                         0.1961                                                         0.2127                                                       0.7842
    matched_iou_repaired                                                              0.1495                                                   0.1163                                                   0.1403                                                 0.0238                                                  0.1531                                                     0.1545                                                     0.1608                                                   0.0435                                                    0.1922                                                        0.1835                                                        0.1983                                                       0.1937                                                  0.1850                                                       0.2011                                                     0.3420                                                  0.1740                                                  0.1667                                                    0.1936                                                       0.2382                                                         0.2200                                                         0.2153                                                       0.4205                                                         0.2054                                                         0.2297                                                       0.7874
    

### Repaired: JSON Valid / App Match / Activity Match


```python
# Strict vs. repaired side by side -- the same function as above, just with
# both columns in one call.

plot_bool_grouped(
    df,
    ["json_valid", "json_valid_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="Share of parseable predictions (%)",
    title='JSON Valid: strict vs. repaired, grouped by "dim"',
)

plot_bool_grouped(
    df,
    ["app_match", "app_match_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="App Match (%)",
    title='App Match: strict vs. repaired, grouped by "dim"',
)

plot_bool_grouped(
    df,
    ["activity_match", "activity_match_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="Activity Match (%)",
    title='Activity Match: strict vs. repaired, grouped by "dim"',
)
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    


### Boxplot: Class Accuracy (strict vs. repaired)


```python
show_stats('class_accuracy')
show_stats('class_accuracy_repaired')
```


    
![png](output_28_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.214169   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.176336   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.197940   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.075162   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.211590   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.225525   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.218385   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.112200   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.299063   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.284036   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.280449   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.289396   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.277462   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.301069   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.358193   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.246897   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.249570   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.286904   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.347687   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.331853   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.312616   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.495711   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.316945   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.367260   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.864024   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.310913  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.269147  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.295365  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.125134  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.319237  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.315550  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.327156  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.184940  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.325967  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.322802  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.330174  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.334445  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.325848  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.330129  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.467874  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.317426  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.310134  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.332798  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.374824  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.341487  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.343292  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.456660  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.336960  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.356803  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.292326  0.0   
    
                                                                      25%     50%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0000  0.0408   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0000  0.0526   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0000  0.2000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0000  0.1765   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0000  0.1538   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0000  0.1667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0000  0.1579   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0000  0.2000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0000  0.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0000  0.1111   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.1250   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0000  0.1667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0000  0.2273   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0000  0.2400   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0000  0.2000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0000  0.4231   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0000  0.2121   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0000  0.2857   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.9333  1.0000   
    
                                                                      75%     max  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.3333  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.2500  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.3056  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.1071  0.6173   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3333  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.3500  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3333  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.1944  0.8750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.4583  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.4615  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.4615  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.5000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.3913  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.4000  1.0000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.5000  1.0000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.6667  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.5455  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.5000  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.5000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.6250  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    
                                                                   median     IQR  
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.3333  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0408  0.2500  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.3056  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.1071  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.3333  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0526  0.3500  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.3333  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.1944  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2000  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.1765  0.4583  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.1538  0.4615  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.1667  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.1579  0.4615  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2000  0.5000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0000  1.0000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.1111  0.3913  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.1250  0.4000  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.1667  0.5000  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2273  0.6667  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2400  0.5455  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2000  0.5000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.4231  1.0000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2121  0.5000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2857  0.6250  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.0667  
    


    
![png](output_28_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.293546   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.212004   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.271772   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.101126   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.278803   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.284763   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.279441   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.203911   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.333195   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.322797   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.338327   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.341095   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.335288   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.351973   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.385916   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.296705   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.293853   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.341475   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.464166   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.385670   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.380846   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.516048   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.370489   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.449470   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.874064   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.314739  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.280025  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.300752  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.147938  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.322401  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.316487  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.330309  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.220629  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.329557  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.328538  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.328406  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.328880  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.323150  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.325087  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.458596  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.318828  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.306153  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.330248  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.329167  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.331143  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.332853  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.444375  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.329593  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.334920  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.270492  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.027778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.043478   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.032258   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.033333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.062500   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.034483   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.200000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.100000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.093750   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.083333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.166667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.200000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.105263   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.181818   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.021053   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.166667   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.187500   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.153846   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.140845   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.230769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.274510   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.036364   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.250000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.411765   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.320000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.307692   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.291667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.392857   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.476190   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.309524   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.416667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.169811   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.450000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.450000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.466667   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.344828   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.523810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.538462   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.473684   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.448276   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.533333   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.714286   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.600000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.600000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.571429   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.705882   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.888889   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.200000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.105263   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.181818   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.021053   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.166667   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.187500   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.153846   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.140845   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.230769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.274510   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.036364   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.250000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.411765   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.320000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.307692   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.291667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.392857   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.476190  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.309524  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.416667  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.169811  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.450000  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.450000  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.466667  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.344828  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.472222  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.500000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.456522  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.491551  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.466667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.475962  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.473684  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.448276  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.498851  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.514286  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.500000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.506250  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.488095  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.539216  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    

### Boxplot: Average Bounds IoU (strict vs. repaired)


```python
show_stats('average_bounds_iou')
show_stats('average_bounds_iou_repaired')
```


    
![png](output_30_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.147447   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.139067   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.139051   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.026045   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.154922   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.158880   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.165071   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.039916   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.191700   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.181409   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.189034   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.188953   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.175004   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.193142   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.594613   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.167591   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.161541   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.184711   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.211662   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.211403   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.203607   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.538733   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.203827   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.217131   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.836202   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.275435  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.250852  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.263060  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.041466  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.286566  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.283856  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.298051  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.079156  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.288102  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.284026  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.289884  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.294379  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.282554  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.290274  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.463120  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.281591  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.274752  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.289390  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.314504  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.301569  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.301400  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.443979  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.296571  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.307106  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.294575  0.0   
    
                                                                      25%     50%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0000  0.0180   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0030  0.0410   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0000  0.0199   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0000  0.0094   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0000  0.0133   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0000  0.0305   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0000  0.0180   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0000  0.0001   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0084  0.0710   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0037  0.0625   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0000  0.0652   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0000  0.0604   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0000  0.0542   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0023  0.0708   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0071  0.9175   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0000  0.0450   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.0441   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0000  0.0629   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0000  0.0642   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0047  0.0833   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0000  0.0718   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0248  0.6013   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0043  0.0792   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0000  0.0867   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.8080  1.0000   
    
                                                                      75%     max  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.1367  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1289  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.1308  1.0000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0326  0.3926   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.1454  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.1508  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.1522  1.0000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0481  0.7958   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2043  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.1892  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.2067  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.2041  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.1823  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2105  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.1658  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.1572  1.0000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.1958  1.0000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2648  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2443  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2341  1.0000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2265  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2552  1.0000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  1.0000   
    
                                                                   median     IQR  
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0180  0.1367  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0410  0.1259  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0199  0.1308  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0094  0.0326  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0133  0.1454  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0305  0.1508  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0180  0.1522  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0001  0.0481  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0710  0.1959  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0625  0.1855  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0652  0.2067  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0604  0.2041  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0542  0.1823  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0708  0.2082  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.9175  0.9929  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0450  0.1658  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0441  0.1572  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0629  0.1958  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0642  0.2648  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0833  0.2396  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0718  0.2341  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.6013  0.9752  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0792  0.2222  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0867  0.2552  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.1920  
    


    
![png](output_30_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.195303   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.160936   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.179033   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.049432   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.201387   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.198113   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.213079   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.076396   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.225243   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.217325   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.229069   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.225083   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.209552   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.226458   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.609421   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.200607   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.186778   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.225499   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.267443   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.246968   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.247780   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.548352   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.237651   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.260987   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.840638   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.281755  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.257546  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.266517  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.069232  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.286848  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.286385  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.297325  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.112641  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.297018  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.292418  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.291806  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.295048  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.283832  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.290900  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.448479  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.283918  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.274881  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.292843  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.300659  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.299370  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.298725  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.436001  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.295593  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.303631  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.284098  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.021586   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.018133   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.021160   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.008416   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.020583   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.020833   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.022629   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.009921   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.031108   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.027464   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.035316   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.031908   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.028348   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.036868   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.043697   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.025463   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.021998   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.035913   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.062814   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.044567   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.044067   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.049354   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.043568   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.052387   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.808019   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.075556   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.058824   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.069265   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.027056   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.078375   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.073304   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.080272   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.035907   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.097093   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.090514   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.104283   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.097284   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.089115   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.102679   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.917526   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.076673   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.071475   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.100755   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.144311   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.119496   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.121560   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.618056   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.113642   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.133347   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.211891   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.153875   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.187870   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.064886   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.222859   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.215823   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.245544   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.097661   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.260338   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.254373   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.280563   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.268549   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.241919   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.269953   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.224397   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.197776   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.262057   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.334032   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.311688   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.314879   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.285714   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.333333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.969697   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.075556   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.058824   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.069265   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.027056   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.078375   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.073304   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.080272   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.035907   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.097093   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.090514   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.104283   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.097284   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.089115   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.102679   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.917526   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.076673   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.071475   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.100755   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.144311   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.119496   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.121560   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.618056   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.113642   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.133347   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.190305  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.135743  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.166710  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.056470  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.202276  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.194990  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.222915  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.087741  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.229231  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.226908  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.245247  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.236641  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.213571  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.233084  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.956303  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.198934  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.175777  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.226143  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.271218  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.267121  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.270812  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.950646  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.242146  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.280947  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.191981  
    

### Boxplot: Levenshtein Similarity (strict vs. repaired)


```python
show_stats('levenshtein_similarity')
show_stats('levenshtein_similarity_repaired')
```


    
![png](output_32_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.413917   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.427867   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.411802   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.303935   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.399773   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.416106   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.407394   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.319245   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.495116   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.479759   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.489895   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.488432   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.468967   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.503300   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.646843   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.471061   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.470462   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.488462   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.577837   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.514561   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.495973   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.680846   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.509466   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.512298   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.928115   
    
                                                                        std  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.234545   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.200513   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.227083   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.115377   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.250143   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.239948   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.250743   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.154006   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.217904   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.224111   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.228593   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.228797   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.231778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.222899   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.267070   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.227637   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.213275   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.225761   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.193719   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.227838   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.238962   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.247411   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.224579   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.239163   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.134474   
    
                                                                      min     25%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.0037  0.2185   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.0007  0.3221   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.0021  0.2233   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.0042  0.2167   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.0016  0.1882   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.0015  0.2301   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.0033  0.1994   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.0009  0.1994   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.0012  0.3672   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.0005  0.3406   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.0001  0.3582   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.0041  0.3336   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.0026  0.3203   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.0022  0.3795   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.0052  0.4536   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.0012  0.3178   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.0000  0.3522   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.0010  0.3474   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.0093  0.4590   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.0006  0.3799   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.0028  0.3340   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.0001  0.4979   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.0000  0.3783   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.0013  0.3537   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.0000  0.9237   
    
                                                                      50%     75%  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.4152  0.5401   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.4338  0.5133   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.4207  0.5339   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.3224  0.3837   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3869  0.5324   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.4122  0.5295   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3958  0.5362   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.3148  0.4392   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.4877  0.5966   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.4809  0.5900   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.4925  0.6053   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.4857  0.6028   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.4738  0.5879   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.5032  0.6149   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.6057  1.0000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.4756  0.5851   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.4690  0.5668   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.4873  0.6020   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.5569  0.6742   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.5065  0.6318   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.4986  0.6279   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.6673  0.9269   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.4984  0.6200   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.5129  0.6495   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.9729  1.0000   
    
                                                                      max  median  \
    dim                                                                             
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.0000  0.4152   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.0000  0.4338   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.0000  0.4207   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.7410  0.3224   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.0000  0.3869   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.0000  0.4122   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.0000  0.3958   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.7692  0.3148   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.0000  0.4877   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.0000  0.4809   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.0000  0.4925   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.0000  0.4857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.0000  0.4738   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.0000  0.5032   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.0000  0.6057   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.0000  0.4756   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.0000  0.4690   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.0000  0.4873   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.0000  0.5569   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.0000  0.5065   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.0000  0.4986   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.0000  0.6673   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.0000  0.4984   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.0000  0.5129   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.0000  0.9729   
    
                                                                      IQR  
    dim                                                                    
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.3216  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.1912  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.3106  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.1670  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.3442  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.2994  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.3368  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.2398  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.2294  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.2494  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.2471  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.2692  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.2676  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.2354  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.5464  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.2673  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.2146  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.2546  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.2152  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.2519  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.2939  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.4290  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.2417  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.2958  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.0763  
    


    
![png](output_32_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.421160   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.424659   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.415440   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.286648   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.408785   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.422254   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.418469   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.256078   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.486357   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.470036   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.487320   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.484571   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.469813   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.498073   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.628783   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.471807   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.475421   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.484611   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.562226   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.512677   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.499924   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.668510   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.506485   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.520678   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.926825   
    
                                                                        std  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.235352   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.202581   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.229965   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.120552   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.247716   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.237228   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.247169   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.153214   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.227114   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.233332   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.233315   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.233759   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.232478   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.229005   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.288335   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.230375   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.211977   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.230718   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.210155   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.231119   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.237398   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.266297   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.228021   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.234793   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.138169   
    
                                                                        min  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.003692   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000439   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.002170   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.004212   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.001735   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.001481   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.002307   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000917   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.001170   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000486   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000122   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.002124   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.002813   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.002382   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.004007   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.001232   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.001106   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.009428   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000635   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.002955   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000143   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.001276   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.000000   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.231734   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.316146   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.232165   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.196479   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.201070   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.252041   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.223732   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.131595   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.358132   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.326972   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.368893   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.340858   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.334847   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.377503   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.426877   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.345011   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.370311   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.350442   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.439584   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.384913   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.360577   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.491935   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.381935   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.386209   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.922939   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.430450   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.431907   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.431685   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.295070   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.404648   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.418768   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.413043   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.248576   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.484574   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.476433   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.492387   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.484169   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.474371   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.502119   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.598985   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.478359   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.474158   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.486891   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.550197   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.506833   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.501794   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.667814   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.496294   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.519106   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.972743   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.544634   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.511732   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.537117   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.375462   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.535891   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.531026   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.539147   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.359021   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.595238   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.588189   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.604621   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.600746   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.587750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.613335   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.584961   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.568733   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.600187   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.671604   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.631213   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.627432   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.926910   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.617664   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.649871   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.741290   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.665472   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.430450   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.431907   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.431685   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.295070   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.404648   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.418768   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.413043   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.248576   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.484574   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.476433   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.492387   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.484169   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.474371   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.502119   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.598985   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.478359   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.474158   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.486891   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.550197   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.506833   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.501794   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.667814   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.496294   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.519106   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.972743   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.312900  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.195586  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.304952  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.178984  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.334821  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.278985  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.315415  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.227426  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.237106  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.261217  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.235728  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.259888  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.252903  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.235833  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.573123  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.239950  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.198422  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.249745  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.232020  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.246300  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.266855  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.434975  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.235730  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.263661  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.077061  
    

## Order-free component metrics

`class_accuracy` and `average_bounds_iou` pair components by their position in
the flattened DFS order. Two siblings emitted in the other order score zero
twice, and a single component too many near the top shifts *every* following
pair by one. Neither says anything about the quality of the screen: position
lives in `bounds`, not in the order of the list.

The metrics below drop that pairing.

| column | what it measures |
|---|---|
| `class_unordered_recall` / `_precision` / `_f1` | class multiset overlap over the whole tree -- order and nesting ignored entirely |
| `class_level_recall` / `_precision` / `_f1` | the same overlap per tree depth -- free within a level, the depth has to be right |
| `matched_iou` | greedy best-first IoU matching of same-class components, over `max(n_pred, n_gt)` |

Recall is against the ground truth, precision against the prediction. Recall
alone would reward a model that simply emits one of every class it knows,
which is why all three are carried.

Every value is a **per-screen** score; the aggregation across screens happens
here, in the cells below. Each metric appears twice: bare for the strict parse
(unparseable prediction = 0.0) and with `_repaired` for the json_repair'd and
loop-collapsed one, the same split the columns above use.


```python
# `df` holds the report values as objects, so every order-free column is coerced
# once here rather than in each cell below. The index is kept so plot_bool_grouped()
# and show_stats() can still address "dim" as an index level.

ORDER_FREE_COLS = [
    col
    for key in UNORDERED_METRIC_KEYS
    for col in (key, f"{key}_repaired")
]


def numeric_view(columns):
    """A float-typed copy of `df[columns]`, index intact."""
    return df[columns].apply(pd.to_numeric, errors="coerce")


by_dim = df.index.get_level_values("dim")
order_free = numeric_view(ORDER_FREE_COLS)

for stat in ("mean", "median"):
    table = getattr(order_free.groupby(by_dim), stat)().rename_axis("dim")
    table = table.reindex(sorted(table.index, key=first_number_sort_key))
    print(f"=== per-screen {stat} ===")
    print(table.T.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
```

    === per-screen mean ===
    dim                                 4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b  4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base  4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b  4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base  4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base  4096_last__via__embeddings_4096_last_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b  8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b  128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base
    class_unordered_recall                                                            0.3031                                                   0.2682                                                   0.2948                                                 0.1063                                                  0.2817                                                     0.3014                                                     0.2870                                                   0.1129                                                    0.4122                                                        0.3936                                                        0.3959                                                       0.3940                                                  0.3831                                                       0.4171                                                     0.3682                                                  0.3716                                                  0.3819                                                    0.3901                                                       0.4164                                                         0.4425                                                         0.4188                                                       0.5338                                                         0.4014                                                         0.4726                                                       0.8690
    class_unordered_recall_repaired                                                   0.3738                                                   0.2988                                                   0.3658                                                 0.1329                                                  0.3443                                                     0.3536                                                     0.3434                                                   0.1919                                                    0.4304                                                        0.4117                                                        0.4363                                                       0.4324                                                  0.4285                                                       0.4543                                                     0.3891                                                  0.4115                                                  0.4248                                                    0.4339                                                       0.5291                                                         0.4809                                                         0.4700                                                       0.5513                                                         0.4371                                                         0.5343                                                       0.8805
    class_unordered_precision                                                         0.2703                                                   0.2511                                                   0.2758                                                 0.2164                                                  0.2759                                                     0.3001                                                     0.2839                                                   0.2420                                                    0.3965                                                        0.3905                                                        0.4116                                                       0.3759                                                  0.3705                                                       0.4003                                                     0.3796                                                  0.3619                                                  0.3498                                                    0.3679                                                       0.4070                                                         0.4120                                                         0.3979                                                       0.5361                                                         0.3859                                                         0.4222                                                       0.8705
    class_unordered_precision_repaired                                                0.4099                                                   0.3066                                                   0.3920                                                 0.2447                                                  0.3768                                                     0.3908                                                     0.3818                                                   0.4342                                                    0.4752                                                        0.4884                                                        0.5261                                                       0.4741                                                  0.4751                                                       0.4992                                                     0.4347                                                  0.4648                                                  0.4255                                                    0.4651                                                       0.6042                                                         0.5008                                                         0.5102                                                       0.5788                                                         0.4718                                                         0.5344                                                       0.8893
    class_unordered_f1                                                                0.2791                                                   0.2516                                                   0.2781                                                 0.1388                                                  0.2693                                                     0.2884                                                     0.2756                                                   0.1380                                                    0.3939                                                        0.3805                                                        0.3966                                                       0.3766                                                  0.3672                                                       0.4015                                                     0.3702                                                  0.3604                                                  0.3591                                                    0.3723                                                       0.4078                                                         0.4186                                                         0.4002                                                       0.5303                                                         0.3874                                                         0.4371                                                       0.8696
    class_unordered_f1_repaired                                                       0.3479                                                   0.2781                                                   0.3390                                                 0.1571                                                  0.3220                                                     0.3338                                                     0.3255                                                   0.2257                                                    0.4247                                                        0.4140                                                        0.4445                                                       0.4208                                                  0.4180                                                       0.4473                                                     0.3937                                                  0.4059                                                  0.4020                                                    0.4224                                                       0.5415                                                         0.4638                                                         0.4588                                                       0.5516                                                         0.4298                                                         0.5038                                                       0.8828
    class_level_recall                                                                0.2414                                                   0.1887                                                   0.2247                                                 0.0557                                                  0.2339                                                     0.2380                                                     0.2396                                                   0.0662                                                    0.3325                                                        0.3168                                                        0.3282                                                       0.3222                                                  0.3116                                                       0.3459                                                     0.3558                                                  0.2984                                                  0.2969                                                    0.3249                                                       0.3806                                                         0.3794                                                         0.3588                                                       0.5151                                                         0.3434                                                         0.4155                                                       0.8637
    class_level_recall_repaired                                                       0.2783                                                   0.2029                                                   0.2589                                                 0.0631                                                  0.2693                                                     0.2645                                                     0.2710                                                   0.0978                                                    0.3454                                                        0.3302                                                        0.3538                                                       0.3416                                                  0.3396                                                       0.3703                                                     0.3636                                                  0.3197                                                  0.3184                                                    0.3527                                                       0.4608                                                         0.4041                                                         0.3923                                                       0.5241                                                         0.3653                                                         0.4592                                                       0.8711
    class_level_precision                                                             0.2173                                                   0.1786                                                   0.2106                                                 0.1164                                                  0.2273                                                     0.2334                                                     0.2349                                                   0.1399                                                    0.3221                                                        0.3142                                                        0.3375                                                       0.3069                                                  0.2980                                                       0.3331                                                     0.3599                                                  0.2920                                                  0.2749                                                    0.3078                                                       0.3709                                                         0.3533                                                         0.3414                                                       0.5128                                                         0.3298                                                         0.3728                                                       0.8647
    class_level_precision_repaired                                                    0.3027                                                   0.2146                                                   0.2774                                                 0.1258                                                  0.2898                                                     0.2865                                                     0.2933                                                   0.2314                                                    0.3793                                                        0.3839                                                        0.4165                                                       0.3693                                                  0.3676                                                       0.4037                                                     0.3866                                                  0.3543                                                  0.3195                                                    0.3774                                                       0.5139                                                         0.4195                                                         0.4236                                                       0.5373                                                         0.3912                                                         0.4597                                                       0.8780
    class_level_f1                                                                    0.2241                                                   0.1790                                                   0.2129                                                 0.0736                                                  0.2238                                                     0.2275                                                     0.2302                                                   0.0804                                                    0.3205                                                        0.3080                                                        0.3278                                                       0.3090                                                  0.2981                                                       0.3344                                                     0.3557                                                  0.2910                                                  0.2815                                                    0.3116                                                       0.3724                                                         0.3598                                                         0.3441                                                       0.5107                                                         0.3321                                                         0.3858                                                       0.8640
    class_level_f1_repaired                                                           0.2634                                                   0.1940                                                   0.2451                                                 0.0781                                                  0.2551                                                     0.2525                                                     0.2585                                                   0.1159                                                    0.3426                                                        0.3323                                                        0.3599                                                       0.3341                                                  0.3310                                                       0.3659                                                     0.3650                                                  0.3176                                                  0.3053                                                    0.3453                                                       0.4677                                                         0.3910                                                         0.3848                                                       0.5218                                                         0.3601                                                         0.4354                                                       0.8729
    matched_iou                                                                       0.1356                                                   0.1105                                                   0.1282                                                 0.0228                                                  0.1418                                                     0.1455                                                     0.1507                                                   0.0309                                                    0.1841                                                        0.1753                                                        0.1883                                                       0.1853                                                  0.1745                                                       0.1908                                                     0.3394                                                  0.1658                                                  0.1585                                                    0.1807                                                       0.2052                                                         0.2098                                                         0.2013                                                       0.4174                                                         0.1961                                                         0.2127                                                       0.7842
    matched_iou_repaired                                                              0.1495                                                   0.1163                                                   0.1403                                                 0.0238                                                  0.1531                                                     0.1545                                                     0.1608                                                   0.0435                                                    0.1922                                                        0.1835                                                        0.1983                                                       0.1937                                                  0.1850                                                       0.2011                                                     0.3420                                                  0.1740                                                  0.1667                                                    0.1936                                                       0.2382                                                         0.2200                                                         0.2153                                                       0.4205                                                         0.2054                                                         0.2297                                                       0.7874
    
    === per-screen median ===
    dim                                 4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b  4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base  4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base  4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b  4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base  4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base  4096_last__via__embeddings_4096_last_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new  4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b  4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b  8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b  128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base
    class_unordered_recall                                                            0.1333                                                   0.1556                                                   0.1429                                                 0.0168                                                  0.0588                                                     0.1667                                                     0.0811                                                   0.0000                                                    0.4000                                                        0.3750                                                        0.3824                                                       0.3750                                                  0.3636                                                       0.4211                                                     0.0000                                                  0.3333                                                  0.3571                                                    0.3750                                                       0.4000                                                         0.4500                                                         0.4146                                                       0.5952                                                         0.3704                                                         0.5000                                                       1.0000
    class_unordered_recall_repaired                                                   0.3333                                                   0.2179                                                   0.3333                                                 0.0485                                                  0.2609                                                     0.2903                                                     0.2500                                                   0.1368                                                    0.4151                                                        0.3889                                                        0.4194                                                       0.4118                                                  0.4130                                                       0.4500                                                     0.0633                                                  0.3871                                                  0.4000                                                    0.4286                                                       0.5294                                                         0.4800                                                         0.4615                                                       0.6000                                                         0.4000                                                         0.5532                                                       1.0000
    class_unordered_precision                                                         0.0909                                                   0.1176                                                   0.1034                                                 0.0366                                                  0.0526                                                     0.1364                                                     0.0667                                                   0.0000                                                    0.3438                                                        0.3333                                                        0.3846                                                       0.3182                                                  0.3030                                                       0.3684                                                     0.0000                                                  0.2941                                                  0.2941                                                    0.3000                                                       0.3750                                                         0.3750                                                         0.3488                                                       0.6000                                                         0.3333                                                         0.4000                                                       1.0000
    class_unordered_precision_repaired                                                0.3571                                                   0.2000                                                   0.3333                                                 0.0980                                                  0.2727                                                     0.3043                                                     0.2857                                                   0.4167                                                    0.4737                                                        0.5000                                                        0.5294                                                       0.4762                                                  0.4762                                                       0.5000                                                     0.1579                                                  0.4615                                                  0.4000                                                    0.4545                                                       0.6389                                                         0.5000                                                         0.5000                                                       0.6765                                                         0.4500                                                         0.5333                                                       1.0000
    class_unordered_f1                                                                0.1081                                                   0.1270                                                   0.1176                                                 0.0256                                                  0.0556                                                     0.1429                                                     0.0741                                                   0.0000                                                    0.3704                                                        0.3529                                                        0.3860                                                       0.3333                                                  0.3333                                                       0.4000                                                     0.0000                                                  0.3077                                                  0.3226                                                    0.3333                                                       0.3913                                                         0.4068                                                         0.3784                                                       0.5926                                                         0.3415                                                         0.4444                                                       1.0000
    class_unordered_f1_repaired                                                       0.2778                                                   0.1818                                                   0.2667                                                 0.0580                                                  0.2222                                                     0.2500                                                     0.2222                                                   0.1714                                                    0.4000                                                        0.3908                                                        0.4333                                                       0.3944                                                  0.4000                                                       0.4444                                                     0.0755                                                  0.3750                                                  0.3750                                                    0.4000                                                       0.5556                                                         0.4516                                                         0.4444                                                       0.6000                                                         0.4000                                                         0.5000                                                       1.0000
    class_level_recall                                                                0.0000                                                   0.0476                                                   0.0000                                                 0.0000                                                  0.0000                                                     0.0625                                                     0.0000                                                   0.0000                                                    0.2143                                                        0.1818                                                        0.2000                                                       0.1786                                                  0.1667                                                       0.2353                                                     0.0000                                                  0.1429                                                  0.1538                                                    0.1944                                                       0.2727                                                         0.2857                                                         0.2500                                                       0.5000                                                         0.2222                                                         0.3462                                                       1.0000
    class_level_recall_repaired                                                       0.1373                                                   0.0769                                                   0.1250                                                 0.0131                                                  0.1154                                                     0.1212                                                     0.1111                                                   0.0280                                                    0.2344                                                        0.2024                                                        0.2500                                                       0.2143                                                  0.2222                                                       0.2778                                                     0.0168                                                  0.1905                                                  0.2000                                                    0.2500                                                       0.4091                                                         0.3158                                                         0.3000                                                       0.5000                                                         0.2500                                                         0.4074                                                       1.0000
    class_level_precision                                                             0.0000                                                   0.0357                                                   0.0000                                                 0.0000                                                  0.0000                                                     0.0556                                                     0.0000                                                   0.0000                                                    0.1818                                                        0.1667                                                        0.2000                                                       0.1538                                                  0.1429                                                       0.2075                                                     0.0000                                                  0.1250                                                  0.1300                                                    0.1667                                                       0.2593                                                         0.2432                                                         0.2121                                                       0.5000                                                         0.2000                                                         0.2745                                                       1.0000
    class_level_precision_repaired                                                    0.1538                                                   0.0682                                                   0.1250                                                 0.0204                                                  0.1176                                                     0.1250                                                     0.1224                                                   0.0882                                                    0.2759                                                        0.2857                                                        0.3333                                                       0.2626                                                  0.2593                                                       0.3333                                                     0.0273                                                  0.2381                                                  0.2000                                                    0.2778                                                       0.5000                                                         0.3333                                                         0.3333                                                       0.5882                                                         0.2941                                                         0.4074                                                       1.0000
    class_level_f1                                                                    0.0000                                                   0.0392                                                   0.0000                                                 0.0000                                                  0.0000                                                     0.0571                                                     0.0000                                                   0.0000                                                    0.1923                                                        0.1667                                                        0.2000                                                       0.1600                                                  0.1500                                                       0.2222                                                     0.0000                                                  0.1333                                                  0.1400                                                    0.1739                                                       0.2667                                                         0.2540                                                         0.2222                                                       0.5000                                                         0.2051                                                         0.3038                                                       1.0000
    class_level_f1_repaired                                                           0.1268                                                   0.0690                                                   0.1067                                                 0.0143                                                  0.1026                                                     0.1096                                                     0.1000                                                   0.0420                                                    0.2326                                                        0.2143                                                        0.2581                                                       0.2083                                                  0.2154                                                       0.2737                                                     0.0187                                                  0.1905                                                  0.1846                                                    0.2353                                                       0.4314                                                         0.3022                                                         0.2895                                                       0.5000                                                         0.2500                                                         0.3750                                                       1.0000
    matched_iou                                                                       0.0058                                                   0.0174                                                   0.0088                                                 0.0007                                                  0.0000                                                     0.0266                                                     0.0061                                                   0.0000                                                    0.0710                                                        0.0637                                                        0.0762                                                       0.0611                                                  0.0628                                                       0.0790                                                     0.0000                                                  0.0473                                                  0.0510                                                    0.0632                                                       0.0611                                                         0.0909                                                         0.0816                                                       0.1416                                                         0.0771                                                         0.0956                                                       1.0000
    matched_iou_repaired                                                              0.0471                                                   0.0283                                                   0.0435                                                 0.0045                                                  0.0394                                                     0.0470                                                     0.0405                                                   0.0238                                                    0.0825                                                        0.0762                                                        0.0892                                                       0.0767                                                  0.0798                                                       0.0927                                                     0.0077                                                  0.0627                                                  0.0644                                                    0.0828                                                       0.1258                                                         0.1045                                                         0.1017                                                       0.1433                                                         0.0905                                                         0.1159                                                       1.0000
    
    


```python
# How much of the reported score was lost to the positional pairing alone.
#
# Same predictions, same ground truth -- the only difference is whether the
# comparison insists on the DFS order. `class_unordered_recall` is the directly
# comparable counterpart to `class_accuracy`: both ask "how many of the ground
# truth's components did the prediction get right".

comparison = numeric_view(
    ["class_accuracy", "class_unordered_recall", "class_unordered_f1", "class_level_f1"]
)

plot_bool_grouped(
    comparison,
    ["class_accuracy", "class_unordered_recall", "class_level_f1"],
    by="dim",
    as_percentage=True,
    xlabel="per-screen mean (%)",
    title="Class score: positional vs. order-free vs. depth-aware (strict)",
)

# Share of screens the positional metric scores *below* the order-free one --
# i.e. screens where it is measuring the ordering, not the reconstruction.
understated = (
    comparison["class_unordered_recall"] > comparison["class_accuracy"] + 1e-9
)
share = (understated.groupby(by_dim).mean() * 100).rename("understated_%")
share = share.reindex(sorted(share.index, key=first_number_sort_key))

print("Screens where the positional pairing understates the reconstruction:")
print(share.to_string(float_format=lambda v: f"{v:6.1f} %"))
```


    
![png](output_35_0.png)
    


    Screens where the positional pairing understates the reconstruction:
    dim
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b              38.2 %
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base           50.4 %
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base           41.1 %
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base             42.9 %
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b            34.5 %
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base         40.8 %
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base         35.2 %
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base           26.1 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b          52.9 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new      51.4 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b      50.4 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base       47.6 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b            48.3 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base       51.7 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base          8.5 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b            50.5 %
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b            54.1 %
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b          46.5 %
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b       34.8 %
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base     51.2 %
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base     48.0 %
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base       15.3 %
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base     46.9 %
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base     47.8 %
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base        6.9 %
    


```python
# The same comparison for the bounding boxes. `average_bounds_iou` pairs by DFS
# position and divides by min(n_pred, n_gt); `matched_iou` matches same-class
# components best-first by IoU and divides by max(n_pred, n_gt), so invented and
# dropped components cost score instead of being free.

plot_bool_grouped(
    numeric_view(["average_bounds_iou", "matched_iou"]),
    ["average_bounds_iou", "matched_iou"],
    by="dim",
    as_percentage=True,
    xlabel="per-screen mean IoU (%)",
    title="Bounds IoU: positional vs. matched (strict)",
)
```


    
![png](output_36_0.png)
    


### Strict vs. repaired


```python
plot_bool_grouped(
    numeric_view(["class_unordered_f1", "class_unordered_f1_repaired"]),
    ["class_unordered_f1", "class_unordered_f1_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="per-screen mean (%)",
    title="Unordered Class F1: strict vs. repaired",
)

plot_bool_grouped(
    numeric_view(["class_level_f1", "class_level_f1_repaired"]),
    ["class_level_f1", "class_level_f1_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="per-screen mean (%)",
    title="Depth-Aware Class F1: strict vs. repaired",
)

plot_bool_grouped(
    numeric_view(["matched_iou", "matched_iou_repaired"]),
    ["matched_iou", "matched_iou_repaired"],
    by="dim",
    as_percentage=True,
    xlabel="per-screen mean IoU (%)",
    title="Matched Bounds IoU: strict vs. repaired",
)
```


    
![png](output_38_0.png)
    



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    


### Boxplot: Unordered Class Recall / Precision / F1


```python
show_stats('class_unordered_recall')
show_stats('class_unordered_recall_repaired')
show_stats('class_unordered_precision')
show_stats('class_unordered_precision_repaired')
show_stats('class_unordered_f1')
show_stats('class_unordered_f1_repaired')
```


    
![png](output_40_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.303104   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.268239   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.294848   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.106261   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.281738   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.301450   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.286998   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.112924   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.412162   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.393606   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.395935   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.393998   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.383079   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.417106   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.368158   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.371568   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.381924   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.390124   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.416400   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.442526   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.418785   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.533779   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.401422   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.472624   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.869044   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.350826  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.303442  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.338801  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.152923  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.350152  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.338574  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.352469  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.180914  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.341081  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.341160  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.350498  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.356758  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.353132  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.351899  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.465195  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.350452  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.344136  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.356502  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.395903  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.360841  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.364368  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.446504  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.353320  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.376276  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.285910  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.030303   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.133333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.155556   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.142857   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.016807   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.058824   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.166667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.081081   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.382353   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.363636   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.421053   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.357143   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.375000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.400000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.450000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.414634   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.595238   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.370370   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.578947   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.457831   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.550000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.190909   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.533333   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.538462   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.538462   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.191919   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.700000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.666667   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.666667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.800000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.750000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.724138   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.809524   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.714286   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.133333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.155556   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.142857   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.016807   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.058824   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.166667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.081081   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.382353   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.363636   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.421053   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.357143   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.375000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.400000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.450000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.414634   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.595238   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.370370   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.578947  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.457831  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.550000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.190909  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.533333  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.538462  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.538462  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.191919  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.636364  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.700000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.666667  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.666667  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.800000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.750000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.724138  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.809524  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    


    
![png](output_40_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.373836   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.298850   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.365763   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.132865   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.344324   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.353634   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.343355   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.191938   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.430389   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.411666   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.436305   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.432416   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.428496   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.454321   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.389137   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.411494   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.424800   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.433947   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.529077   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.480938   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.469989   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.551325   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.437086   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.534315   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.880492   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.321836  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.292674  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.310767  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.164876  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.330327  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.316960  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.335157  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.203563  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.324891  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.324574  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.321837  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.329046  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.325113  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.323171  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.455776  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.325099  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.316518  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.329476  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.321715  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.329704  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.326981  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.432752  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.328578  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.327837  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.260665  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.074074   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.045455   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.076923   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.030769   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.066667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.027778   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.017241   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.125000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.103448   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.136364   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.120000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.118644   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.100000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.136364   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.115385   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.260870   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.181818   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.170732   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.136364   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.217949   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.048544   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.260870   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.290323   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.250000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.136842   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.415094   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.388889   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.419355   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.411765   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.413043   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.450000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.063291   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.387097   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.400000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.428571   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.529412   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.480000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.461538   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.553191   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.600000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.481481   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.583333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.236559   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.571429   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.566667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.578947   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.320388   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.700000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.666667   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.674157   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.800000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.750000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.727273   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.675676   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.812500   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.873494   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.217949   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.048544   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.260870   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.290323   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.250000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.136842   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.415094   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.388889   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.419355   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.411765   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.413043   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.450000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.063291   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.387097   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.400000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.428571   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.529412   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.480000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.461538   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.553191   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.525926  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.436027  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.506410  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.236559  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.540659  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.500000  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.551170  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.303147  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.541667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.563218  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.530303  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.546667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.548023  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.546154  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.566667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.530303  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.558773  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.539130  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.568182  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.556541  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.539312  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.562500  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    


    
![png](output_40_4.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.270251   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.251051   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.275847   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.216357   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.275875   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.300109   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.283914   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.241976   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.396484   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.390454   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.411562   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.375864   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.370524   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.400295   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.379633   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.361923   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.349825   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.367878   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.407043   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.412002   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.397865   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.536117   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.385927   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.422221   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.870537   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.334685  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.301955  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333714  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.284149  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.353829  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.351356  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.358982  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.343813  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.352087  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.357325  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.364772  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.358079  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.356463  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.354002  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.465094  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.354325  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.333546  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.354793  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.391283  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.356408  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.363980  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.445523  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.353054  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.362282  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.284979  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.017544   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.941176   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.090909   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.117647   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.103448   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.036585   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.052632   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.136364   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.066667   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.343750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.384615   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.318182   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.303030   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.368421   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.294118   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.294118   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.300000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.375000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.375000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.348837   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.333333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.500000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.421053   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.500000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.439024   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.521739   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.555556   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.533333   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.525000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.680000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.689655   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.733333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.690476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.650000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.600000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.658537   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.777778   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.714286   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.711864   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.738095   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.984615   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.090909   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.117647   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.103448   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.036585   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.052632   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.136364   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.066667   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.343750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.384615   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.318182   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.303030   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.368421   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.294118   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.294118   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.300000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.375000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.375000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.348837   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.333333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.500000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.421053  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.500000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.439024  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.521739  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.555556  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.533333  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.525000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.662456  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.689655  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.733333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.690476  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.650000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.600000  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.658537  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.777778  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.714286  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.711864  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.738095  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.058824  
    


    
![png](output_40_6.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.409904   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.306621   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.391997   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.244737   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.376791   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.390800   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.381763   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.434191   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.475174   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.488417   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.526134   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.474146   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.475095   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.499227   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.434719   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.464788   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.425497   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.465115   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.604207   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.500806   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.510151   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.578829   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.471789   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.534390   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.889304   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.350336  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.318127  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.345083  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.284398  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.364979  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.356888  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.366353  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.360029  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.346498  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.351996  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.341191  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.344799  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.347116  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.340597  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.459771  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.347722  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.327672  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.344750  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.313651  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.336821  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.342038  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.427141  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.341742  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.330765  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.250308  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.065476   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.030864   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.054054   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.014388   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.050000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.016949   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.065476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.148148   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.150000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.222222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.190476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.129630   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.116564   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.142857   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.368421   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.200000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.200000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.157895   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.952381   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.357143   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.200000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.098039   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.272727   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.304348   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.285714   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.416667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.473684   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.529412   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.476190   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.476190   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.157895   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.461538   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.400000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.454545   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.638889   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.676471   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.450000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.533333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.680000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.500000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.666667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.475000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.666667   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.666667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.680000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.764706   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.777778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.800000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.833333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.764706   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.777778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.800000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.750000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.666667   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.750000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.880000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.800000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.816327   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.750000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.833333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.984615   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.357143   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.200000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.098039   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.272727   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.304348   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.285714   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.416667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.473684   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.529412   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.476190   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.476190   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.500000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.157895   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.461538   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.400000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.454545   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.638889   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.676471   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.450000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.533333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.614524  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.469136  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.612613  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.475000  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.652278  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.616667  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.663051  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.699230  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.629630  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.650000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.611111  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.621849  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.634921  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.609524  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.620370  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.550102  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.607143  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.511579  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.600000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.616327  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.592105  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.583333  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.047619  
    


    
![png](output_40_8.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.279128   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.251553   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.278095   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.138824   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.269302   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.288424   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.275603   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.138031   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.393925   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.380481   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.396559   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.376590   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.367190   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.401521   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.370164   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.360420   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.359141   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.372327   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.407763   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.418597   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.400180   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.530294   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.387434   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.437088   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.869553   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.334187  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.295634  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.328415  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.191651  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.339941  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.331766  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.343763  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.202927  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.338285  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.339895  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.349426  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.350779  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.344994  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.345979  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.462822  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.346494  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.332190  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.348615  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.388295  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.351228  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.356784  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.443425  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.347386  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.360736  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.285389  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.022989   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.108108   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.126984   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.117647   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.025641   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.055556   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.142857   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.074074   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.370370   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.352941   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.385965   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.307692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.322581   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.333333   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.391304   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.406780   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.378378   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.592593   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.341463   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.444444   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.523077   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.428571   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.521739   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.258065   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.509434   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.520548   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.521739   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.285714   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.651163   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.676056   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.642857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.638298   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.615385   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.653061   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.774194   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.705882   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.695122   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.750000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.769231   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.873096   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.108108   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.126984   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.117647   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.025641   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.055556   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.142857   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.074074   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.370370   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.352941   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.385965   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.307692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.322581   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.333333   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.391304   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.406780   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.378378   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.592593   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.341463   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.444444   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.523077  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.428571  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.521739  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.258065  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.509434  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.520548  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.521739  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.285714  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.643678  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.651163  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.676056  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.642857  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.666667  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.638298  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.615385  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.653061  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.774194  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.705882  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.695122  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.750000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    


    
![png](output_40_10.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.347880   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.278054   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.338960   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.157149   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.322013   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.333809   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.325504   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.225700   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.424688   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.414028   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.444520   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.420756   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.417995   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.447279   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.393742   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.405895   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.401996   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.422366   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.541537   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.463752   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.458838   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.551629   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.429827   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.503850   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.882780   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.310842  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.286371  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.306345  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.190462  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.322764  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.313582  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.327078  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.213803  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.320621  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.320133  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.318142  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.323743  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.318095  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.318074  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.452874  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.321624  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.309634  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.323233  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.307430  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.320624  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.320638  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.427424  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.322737  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.315783  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.257239  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.068966   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.037037   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.062500   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.022472   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.055556   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.024691   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.028571   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.129032   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.117647   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.123894   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.117647   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.105263   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.117647   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.121212   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.300000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.180000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.172589   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.140351   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.232558   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.277778   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.181818   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.266667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.057971   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.222222   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.250000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.222222   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.171429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.390805   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.433333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.394366   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.444444   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.075472   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.375000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.400000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.555556   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.451613   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.444444   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.571429   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.444444   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.555556   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.293706   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.545455   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.545455   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.551724   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.692308   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.690599   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.631579   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.666667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.800000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.714286   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.711111   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.761905   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.884146   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.914062   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.277778   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.181818   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.266667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.057971   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.222222   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.250000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.222222   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.171429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.390805   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.433333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.394366   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.400000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.444444   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.075472   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.375000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.375000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.400000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.555556   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.451613   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.444444   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.400000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.502463  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.407407  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.493056  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.293706  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.522983  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.489899  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.527033  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.371429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.537634  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.549020  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.538462  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.542773  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.549020  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.536753  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.561404  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.513932  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.545455  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.500000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.534286  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.538522  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.526316  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.529347  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    

### Boxplot: Depth-Aware Class Recall / Precision / F1


```python
show_stats('class_level_recall')
show_stats('class_level_recall_repaired')
show_stats('class_level_precision')
show_stats('class_level_precision_repaired')
show_stats('class_level_f1')
show_stats('class_level_f1_repaired')
```


    
![png](output_42_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.241431   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.188700   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.224745   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.055742   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.233913   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.238017   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.239642   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.066183   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.332489   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.316803   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.328213   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.322222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.311551   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.345917   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.355802   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.298447   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.296895   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.324933   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.380630   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.379369   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.358807   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.515107   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.343439   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.415465   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.863698   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.334954  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.286352  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.321427  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.098513  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.337443  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.327971  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.341750  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.135005  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.344297  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.343216  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.350095  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.355619  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.350054  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.351694  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.468296  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.345263  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.338715  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.352702  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.390815  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.363491  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.363424  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.455361  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.353431  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.377529  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.292876  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.928571   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.047619   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.062500   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.181818   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.178571   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.235294   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.153846   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.194444   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.272727   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.285714   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.250000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.222222   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.346154   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.428571   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.250000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.375000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.073529   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.400000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.388889   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.400000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.072165   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.588235   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.600000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.583333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.618398   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.545455   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.521739   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.600000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.750000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.769231   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.673267   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.047619   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.062500   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.181818   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.178571   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.235294   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.153846   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.194444   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.272727   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.285714   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.250000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.222222   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.346154   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.428571  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.250000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.375000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.073529  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.400000  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.388889  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.400000  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.072165  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.588235  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.600000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.583333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.571429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.618398  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.545455  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.521739  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.600000  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.750000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.666667  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.666667  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.600000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.769231  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.071429  
    


    
![png](output_42_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.278279   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.202901   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.258876   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.063144   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.269298   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.264506   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.271013   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.097756   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.345421   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.330196   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.353794   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.341554   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.339575   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.370271   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.363611   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.319687   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.318438   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.352675   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.460850   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.404101   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.392281   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.524075   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.365339   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.459240   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.871114   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.323519  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.283669  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.312486  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.102326  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.329053  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.320110  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.334480  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.153204  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.335887  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.334326  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.335807  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.342701  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.337272  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.336833  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.463967  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.334313  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.328920  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.338820  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.343672  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.345506  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.342862  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.448010  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.339450  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.347890  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.275335  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.047619   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.037037   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.056604   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.043478   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.043478   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.066667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.024390   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.031250   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.049180   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.142857   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.093023   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.084746   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.073171   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.137931   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.928571   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.137255   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.076923   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.125000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.013072   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.115385   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.121212   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.111111   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.028037   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.234375   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.202381   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.222222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.277778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.016807   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.190476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.250000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.409091   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.315789   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.300000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.407407   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.466667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.277778   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.411765   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.084906   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.444444   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.416667   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.454545   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.136842   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.593750   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.600000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.583333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.583333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.625000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.555556   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.538462   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.600000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.764706   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.769231   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.673267   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.137255   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.076923   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.125000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.013072   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.115385   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.121212   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.111111   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.028037   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.234375   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.202381   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.250000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.222222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.277778   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.016807   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.190476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.250000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.409091   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.315789   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.300000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.407407   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.466667  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.277778  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.411765  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.084906  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.444444  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.416667  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.454545  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.136842  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.546131  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.534392  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.543396  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.539855  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.539855  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.558333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.531165  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.507212  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.550820  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.621849  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.573643  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.581921  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.526829  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.631300  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.071429  
    


    
![png](output_42_4.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.217313   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.178574   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.210609   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.116374   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.227270   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.233449   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.234933   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.139855   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.322063   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.314208   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.337525   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.306903   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.297981   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.333087   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.359878   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.291986   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.274859   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.307771   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.370935   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.353311   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.341423   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.512817   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.329779   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.372812   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.864725   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.319757  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.283465  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.313214  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.188629  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.336044  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.331694  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.343311  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.244670  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.348826  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.350035  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.357796  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.349987  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.344213  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.350917  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.465226  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.346707  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.327727  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.348906  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.384206  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.354559  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.359494  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.452227  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.349079  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.361513  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.292077  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.035714   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.055556   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.181818   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.207547   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.125000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.130000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.166667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.259259   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.243243   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.212121   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.200000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.274510   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.225806   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.156250   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.363636   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.352941   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.375000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.619048   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.555556   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.533333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.600000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.521739   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.485714   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.551724   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.727273   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.625000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.625000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.578947   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.909091   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.035714   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.055556   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.181818   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.153846   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.142857   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.207547   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.125000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.130000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.166667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.259259   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.243243   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.212121   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.200000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.274510   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.333333  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.225806  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333333  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.156250  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.363636  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.352941  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.375000  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.153846  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.571429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.619048  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.555556  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.533333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.600000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.521739  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.485714  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.551724  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.727273  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.625000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.625000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.578947  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.666667  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    


    
![png](output_42_6.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.302657   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.214647   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.277420   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.125816   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.289758   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.286545   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.293313   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.231406   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.379286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.383915   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.416452   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.369251   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.367598   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.403679   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.386567   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.354261   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.319524   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.377449   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.513916   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.419478   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.423619   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.537336   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.391196   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.459710   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.877961   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.343136  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.304452  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.333361  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.192512  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.350198  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.344489  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.354327  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.282265  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.355222  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.359099  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.356533  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.351826  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.349515  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.352050  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.460880  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.353744  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.331892  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.352256  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.337884  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.348730  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.354266  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.441167  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.347281  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.348047  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.266762  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.046512   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.037500   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.076923   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.040000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.043478   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.071429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.018182   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.019231   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.050000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.214286   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.100000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.095238   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.080000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.142857   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.941176   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.153846   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.068182   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.125000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.020408   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.117647   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.125000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.122449   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.088235   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.275862   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.285714   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.262626   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.259259   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.027273   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.238095   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.277778   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.333333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.333333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.588235   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.294118   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.407407   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.500000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.300000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.500000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.181818   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.500000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.500000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.500000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.404762   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.666667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.722222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.658537   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.636364   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.687500   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.625000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.538462   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.666667   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.818182   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.714286   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.733333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.666667   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.769231   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.909091   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.153846   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.068182   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.125000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.020408   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.117647   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.125000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.122449   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.088235   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.275862   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.285714   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.262626   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.259259   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.333333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.027273   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.238095   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.200000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.277778   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.500000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.333333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.333333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.588235   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.294118   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.407407   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.500000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.300000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.500000  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.181818  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.500000  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.500000  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.500000  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.404762  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.620155  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.629167  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.645299  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.618537  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.592885  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.616071  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.606818  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.519231  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.616667  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.603896  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.614286  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.638095  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.586667  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.626374  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.058824  
    


    
![png](output_42_8.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.224113   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.178999   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.212943   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.073564   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.223796   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.227468   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.230182   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.080429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.320455   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.307982   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.327790   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.308996   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.298137   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.334436   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.355697   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.291030   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.281457   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.311569   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.372400   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.359841   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.344089   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.510715   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.332075   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.385840   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.864037   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.321088  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.280236  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.311770  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.125397  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.328164  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.320998  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.333822  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.145255  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.340719  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.339866  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.348101  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.348202  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.340281  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.346218  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.465587  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.341761  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.328282  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.345725  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.382997  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.353523  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.356035  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.452076  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.347135  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.362568  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.292478  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.933333   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.039216   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.057143   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.192308   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.160000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.150000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.222222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.133333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.140000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.173913   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.266667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.253968   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.222222   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.205128   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.303797   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.372881   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.235294   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.336449   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.099174   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.363636   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.352941   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.375000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.096000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.555556   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.600000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.560000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.538462   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.600000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.533333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.500000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.565217   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.736842   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.645161   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.625000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.588235   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.705882   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.747253   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.873096   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.039216   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.057143   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.192308   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.166667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.200000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.160000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.150000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.222222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.133333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.140000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.173913   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.266667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.253968   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.222222   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.205128   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.303797   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.372881  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.235294  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.336449  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.099174  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.363636  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.352941  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.375000  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.096000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.571429  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.555556  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.600000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.560000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.538462  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.600000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.533333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.500000  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.565217  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.736842  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.645161  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.625000  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.588235  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.705882  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.066667  
    


    
![png](output_42_10.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.263381   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.194029   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.245107   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.078150   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.255142   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.252465   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.258503   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.115937   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.342566   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.332261   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.359874   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.334114   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.330988   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.365872   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.364957   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.317612   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.305291   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.345340   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.467729   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.391027   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.384761   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.521813   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.360057   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.435391   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.872876   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.313097  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.278917  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.305358  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.125665  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.321357  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.315078  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.327054  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.157515  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.332513  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.330806  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.333304  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.336047  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.328697  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.332355  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.460820  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.331857  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.320407  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.333429  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.332015  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.336388  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.336774  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.443423  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.333696  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.335901  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.272668  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.046512   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.038095   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.064516   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.041667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.043243   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.066667   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.022222   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.023952   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.048780   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.166667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.093023   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.085714   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.074074   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.133333   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.928571   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.126829   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.068966   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.106667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.014286   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.102564   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.109589   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.100000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.041958   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.232558   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.258065   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.208333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.215385   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.273684   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.018692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.190476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.184615   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.235294   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.431373   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.302158   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.289474   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.375000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.428571   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.258065   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.387097   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.106870   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.400000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.400000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.416667   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.179310   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.584906   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.615385   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.571429   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.615385   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.553191   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.508475   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.588235   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.757576   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.666667   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.600000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.727273   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.747253   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.873096   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.126829   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.068966   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.106667   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.014286   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.102564   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.109589   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.100000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.041958   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.232558   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.214286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.258065   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.208333   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.215385   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.273684   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.018692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.190476   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.184615   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.235294   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.431373   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.302158   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.289474   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.500000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.250000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.375000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.428571  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.258065  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.387097  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.106870  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.400000  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.400000  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.416667  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.179310  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.538394  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.533333  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.550868  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.529762  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.528185  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.548718  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.530969  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.484522  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.539455  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.590909  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.573643  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.580952  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.525926  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.593939  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.071429  
    

### Boxplot: Matched Bounds IoU


```python
show_stats('matched_iou')
show_stats('matched_iou_repaired')
```


    
![png](output_44_0.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.135598   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.110455   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.128191   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.022776   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.141805   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.145541   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.150740   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.030942   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.184119   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.175304   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.188296   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.185286   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.174488   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.190780   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.339388   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.165769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.158462   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.180745   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.205193   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.209827   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.201250   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.417382   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.196065   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.212658   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.784182   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.257259  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.229816  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.244935  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.038621  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.268400  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.267053  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.279352  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.054056  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.276246  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.270763  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.278040  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.285469  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.270535  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.278411  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.462897  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.270112  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.261197  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.279119  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.302687  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.292824  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.289297  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.450706  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.287700  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.293426  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.321985  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.666667   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.005777   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.017364   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.008844   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000714   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.026605   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.006123   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.070959   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.063692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.076241   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.061102   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.062769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.078993   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.047337   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.051020   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.063246   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.061094   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.090909   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.081563   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.141575   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.077104   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.095580   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.135294   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.098564   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.132955   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.029671   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.140052   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.140670   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.144343   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.041851   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.213831   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.204262   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.224050   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.219275   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.203177   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.229763   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.186780   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.171119   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.211905   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.275260   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.259963   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.252243   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.233504   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.268899   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.282385   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.374858   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.005777   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.017364   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.008844   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000714   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.026605   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.006123   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.070959   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.063692   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.076241   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.061102   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.062769   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.078993   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.047337   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.051020   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.063246   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.061094   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.090909   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.081563   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.141575   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.077104   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.095580   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.135294  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.098564  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.132955  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.029671  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.140052  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.140670  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.144343  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.041851  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.213831  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.204262  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.224050  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.219275  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.203177  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.229763  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.186780  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.171119  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.211905  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.275260  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.259963  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.252243  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.233504  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.268899  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.333333  
    


    
![png](output_44_2.png)
    


                                                                     count  \
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        66261.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base            725.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      66261.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      33733.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base          725.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    66260.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base       2773.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         66261.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         66261.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       66261.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    46927.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  66261.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base     3797.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  66261.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    12757.0   
    
                                                                       mean  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.149546   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.116345   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.140275   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.023774   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.153084   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.154508   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.160845   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.043512   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.192162   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.183463   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.198322   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.193673   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.185035   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.201113   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.341957   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.173955   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.166652   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.193589   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.238199   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.219999   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.215285   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.420510   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.205380   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.229671   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.787388   
    
                                                                        std  min  \
    dim                                                                            
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.254746  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.229795  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.242964  0.0   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.038608  0.0   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.266343  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.264555  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.276671  0.0   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.056798  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.274478  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.269039  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.274711  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.282173  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.267584  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.275156  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.461134  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.267570  0.0   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.259236  0.0   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.276337  0.0   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.289155  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.288337  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.284266  0.0   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.448104  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.284063  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.287783  0.0   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.314464  0.0   
    
                                                                        25%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.000795   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.000548   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.000000   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.000779   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.000676   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.010616   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.006554   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.016290   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.007407   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.010922   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.017462   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.003314   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.006061   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.010132   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.041806   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.025446   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.022316   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.019565   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.034760   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.666667   
    
                                                                        50%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.047148   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.028325   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.043478   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.004504   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.039367   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.047046   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.040476   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.023810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.082479   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.076180   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.089153   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.076729   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.079810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.092749   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.007734   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.062719   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.064389   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.082789   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.125771   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.104466   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.101675   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.143319   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.090502   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.115906   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        75%  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.152103   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.105311   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.145060   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.030667   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.153816   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.150000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.158149   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.061809   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.222880   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.212150   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.233605   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.224932   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.212644   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.239831   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.194606   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.179356   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.226204   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.296910   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.267333   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.264436   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.241074   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.285714   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        max  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        1.000000   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.282385   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      1.000000   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.450980   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         1.000000   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         1.000000   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       1.000000   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  1.000000   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  1.000000   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                     median  \
    dim                                                                       
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.047148   
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.028325   
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.043478   
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.004504   
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.039367   
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.047046   
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.040476   
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.023810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.082479   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.076180   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.089153   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.076729   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.079810   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.092749   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      0.007734   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.062719   
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.064389   
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.082789   
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.125771   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.104466   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.101675   
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    0.143319   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.090502   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.115906   
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    1.000000   
    
                                                                        IQR  
    dim                                                                      
    4096_avg__via__embeddings_4096_avg_full_qwen2.5-1.5b           0.151308  
    4096_avg__via__embeddings_4096_avg_full_qwen3-0.6b-base        0.105311  
    4096_avg__via__embeddings_4096_avg_full_qwen3-1.7b-base        0.144512  
    4096_avg__via__embeddings_4096_avg_full_qwen3-4b-base          0.030667  
    4096_last__via__embeddings_4096_last_full_qwen2.5-1.5b         0.153816  
    4096_last__via__embeddings_4096_last_full_qwen3-0.6b-base      0.149221  
    4096_last__via__embeddings_4096_last_full_qwen3-1.7b-base      0.158149  
    4096_last__via__embeddings_4096_last_full_qwen3-4b-base        0.061133  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b       0.212264  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new   0.205595  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-coder-3b   0.217316  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-0.6b-base    0.217524  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b         0.201723  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-1.7b-base    0.222369  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3-4b-base      1.000000  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-2b         0.191291  
    4x4096_avg__via__embeddings_4x4096_avg_full_qwen3.5-9b         0.173296  
    8x4096_avg__via__embeddings_8x4096_avg_full_qwen2.5-1.5b       0.216072  
    128x4096_avg__via__embeddings_128x4096_avg_50k_qwen2.5-1.5b    0.255104  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-0.6b-base  0.241887  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-1.7b-base  0.242120  
    16x4096_avg__via__embeddings_16x4096_avg_full_qwen3-4b-base    1.000000  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-0.6b-base  0.221509  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-1.7b-base  0.250954  
    64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base    0.333333  
    

## Plots


```python
# Zeilen: Embedding Dimensions
# Spalten: Reconstruction LLM
values = [
    [0.00, 0.00, 0.00, 0.00],  # 4096 (last)
    [0.00, 0.00, 0.00, 0.00],  # 4096 (avg)
    [0.00, 0.00, 0.00, 0.00],  # 4x4096 (avg)
    [0.00, 0.00, 0.00, 0.00],  # 16x4096 (avg)
    [0.00, 0.00, 0.00, 0.00],  # 64x4096 (avg)
]

embedding_dimensions = [
    "4096 (last)",
    "4096 (avg)",
    "4x4096 (avg)",
    "16x4096 (avg)",
    "64x4096 (avg)",
]

reconstruction_llms = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]

df = pd.DataFrame(
    values,
    index=embedding_dimensions,
    columns=reconstruction_llms,
)

fig, ax = plt.subplots(
    figsize=(10, 6),
    facecolor="white",
)

ax.set_facecolor("white")

heatmap = sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Value"},
    ax=ax,
)

heatmap.figure.axes[-1].set_facecolor("white")

ax.set_title("Reconstructions' Median Class Accuracy")
ax.set_xlabel("Reconstruction LLM")
ax.set_ylabel("Embedding Dimensions")

fig.tight_layout()

fig.savefig(
    "median_class_accuracy.pdf",
    bbox_inches="tight",
    facecolor="white",
    transparent=False,
)

fig.savefig(
    "median_class_accuracy.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    transparent=False,
)

plt.show()
```


    
![png](output_46_0.png)
    



```python

```
