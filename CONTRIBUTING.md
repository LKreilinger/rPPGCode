# Contributing

Contributions are welcome: bug reports, fixes, new datasets, new models and better documentation.

## Development setup

```bash
git clone https://github.com/LKreilinger/rPPGCode.git
cd rPPGCode
python -m venv venv
# Windows: venv\Scripts\activate    Linux / macOS: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

You need Python 3.10 to 3.12. You do not need a dataset, a GPU or a Weights & Biases account to run the tests.

## Making a change

1. Open an issue first if the change is large, so the approach can be discussed.
2. Create a branch from `main`.
3. Add or update a test in `tests/` for what you change. `tests/conftest.py` has helpers that write small synthetic UBFC-rPPG, PURE and WCD recordings.
4. Run `pytest`. All tests must pass.
5. Open a pull request and describe what you changed and how you tested it. If you ran the change on a real dataset, say which one.

## License of contributions

By opening a pull request you agree that your contribution is released under the [MIT License](LICENSE) of this project. Only add code from other sources if its license allows it, keep its copyright notice, and list it in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

`cnn_process/load/PhysNet.py` and `cnn_process/load/lossFunction.py` are third-party files for research use only. Keep their headers unchanged.

## Things to know about the code

- **Paths.** Never hard-code a path. Everything is derived from the base folder in `Preprocessing/pre_config.py` and `Preprocessing/config_dataset.py`.
- **Dataset configs.** The first three entries of every dataset config must be `train`, `validation` and `test`. `split_data.py` and the preprocessing loops rely on that order.
- **Naming.** Videos are named `vid_<subject>...` and their pulse data `bvp_<subject>...`. The code finds the pulse data of a video by replacing `vid` with `bvp`. Augmented copies of `vid_s<N>` are named `vid_a_s<N>`.
- **Preprocessed dataset format.** `output/<name>/<split>/<video>/img_00001.jpg ...` plus one `annotations.txt` per split. Each line is `<video> <first frame> <last frame> <128 pulse values>`.
- **Preprocessing changes the raw data folder.** It moves subject folders into `train/`, `validation/` and `test/`. Test with a copy or with the synthetic data from the tests.
- **Results must stay comparable.** If a change alters preprocessing output, the loss or the metrics, say so clearly in the pull request.

## Adding a dataset

1. Add a config to `Preprocessing/config_dataset.py`.
2. Add a preprocessing function. If the dataset consists of videos, `preprocessing_ubfc_main.pre_ubfc` may already work; for image folders, look at `Preprocessing/PURE/`.
3. Register it in `DATASETS` and in `preprocess()` in `__main__.py`.
4. Add a generator for synthetic data to `tests/conftest.py` and a test to `tests/test_preprocessing.py`.
5. Describe the expected folder layout in `README.md`.

## Open tasks

Good first tasks:

- Replace the remaining hard-coded paths in `data_collection/` and the helper scripts with command line arguments.
- Repair or remove `data_collection/__main.py`.
- Move the two standalone MoviePy 1 scripts to the MoviePy 2 API, or remove them.
- Make the Weights & Biases project name configurable.

Larger tasks:

- Describe every experiment in one place (training data, test data, splits), so that one command reproduces it from the raw data.
- Keep only one of the two pulse rate functions (see "Known limitations" in the README) and move it out of `old_py_files/`.
- Copy the three small `pykale` helpers that are used (`VideoFrameDataset`, `ImglistToTensor`, `set_seed`) into the repository. This removes the limits on Python, `torch` and `pandas`.
- Save the best model of a run according to the validation loss, not only the last one.
- Replace the Haar cascade with a more robust face detector and compare the results.
- Remove the duplicated configuration blocks in `cnn_process/train_*.py`.
- Add other rPPG models and compare them with PhysNet.
- Re-run the experiments on the real datasets with the current package versions and publish the numbers.
- Build the Docker image in the automated tests.
