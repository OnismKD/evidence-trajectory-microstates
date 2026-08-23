# Data directory

Raw EEG is not distributed with this repository. Download the public
OpenNeuro `ds004504` example with:

```bash
python scripts/download_ds004504.py --output data/ds004504
```

The default download includes the AD and healthy-control participants used by
the example (`sub-001` through `sub-065`) and their eyes-closed recordings.
The downloader also retrieves the BIDS metadata needed to construct the
manifest.

For non-public or controlled datasets, copy `manifest.example.csv`, populate
one row per recording, and point the relevant YAML file at that manifest. Raw
paths may be absolute or relative to the YAML file.
