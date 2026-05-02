# Shortcuts for AI Image Detection

This project tests whether images can be identified as AI-generated or real based on features extracted from image attributes rather than analyzing the contents of the images themselves.

## Running the code

To run the code in this repository, first install the dependencies with `pip install -r requirements.txt`. Then run the code by running the following command:

```bash
./run.sh
```

All datasets used, by the program, which contain attributes extracted from the original source dataset, should be included in the `data` directory. If you run into any issues, follow the below instructions to download the repository directly.

## Cloning this repository

Before cloning the repository, ensure you have [git-lfs](https://git-lfs.com/) and [git-xet](https://huggingface.co/docs/hub/en/xet/using-xet-storage#git) installed. These are required to download the dataset.

If cloning the repository to make changes, use the SSH address.

```bash
git clone --recurse-submodules git@github.com:samuelstanley0101/AI_Image_Detection.git
```

Otherwise use the HTTPS address. Note that the `--recurse-submodules` flag is removed, which means the dataset is **not** downloaded upon cloning the repository. To download the dataset after cloning, run `git submodule init` and `git submodule update`.

```bash
git clone https://github.com/samuelstanley0101/AI_Image_Detection
```
