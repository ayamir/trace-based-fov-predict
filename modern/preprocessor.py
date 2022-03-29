import re
import os
import cmder

from pathlib import Path
import pandas as pd

SPLIT_PATH = "../../dataset/shanghai_gaze/train_test_set.xlsx"
SET_PATH = "../../dataset/shanghai_gaze/"

DATA = "../../dataset/xr_data/"
DATA1 = "../../dataset/xr_data/Experiment_1/"
DATA2 = "../../dataset/xr_data/Experiment_2/"

HW = 1
PW = 1
FPS = 30
DOWNSAMPLE = 2
MODELS_PATH = "../../models/modern/hw" + str(HW) + "pw" + str(PW) + "models/"


category_dict = {
    "Film": ["2"],
    "Talkshow": ["11", "17"],
    "Documentary": ["5", "8"],
    "Sport": ["1", "4", "7", "12", "13"],
    "Performance": ["0", "3", "6", "9", "10", "14", "15", "16"],
}


def align_filename():
    cmder.infOut("Aligning filename...")
    files = list(Path(DATA2).rglob("video_*.csv"))
    for file in files:
        file = str(file)
        old_no = file[-5:][0]
        new_no = str(eval(old_no) + 9)
        new_file = new_no.join(file.rsplit(old_no, 1))
        os.system(f"mv {file} {new_file}")
    cmder.successOut("Aligning finished.")


def merge_data():
    cmder.infOut("Merging data...")
    files = list(Path(DATA).rglob("video_*.csv"))
    for file in files:
        file = str(file)
        dir = file.rsplit("/", 1)[0]
        new_dir = DATA + dir.rsplit("/", 1)[1]
        os.system(f"mkdir -p {new_dir}")
        os.system(f"mv {file} {new_dir}")
    os.system(f"rm -rf {DATA1} {DATA2}")
    cmder.successOut("Merge finished.")


def classify():
    files = []
    filepaths = list(Path(DATA).rglob("video_*.csv"))
    for file in filepaths:
        file = str(file)
        files.append(file)

    cmder.infOut("Classifying data...")
    for category in category_dict:
        category_dir = DATA + category
        if not os.path.isdir(category_dir):
            os.system(f"mkdir -p {category_dir}")
        nos = category_dict[category]
        category_datas = []
        for no in nos:
            data_name = ".*video_" + no + ".csv"
            r = re.compile(data_name)
            filtered_data = list(filter(r.match, files))
            category_datas += filtered_data
        for category_data in category_datas:
            filename = os.path.basename(category_data)
            # delect table head
            os.system(f"sed -i '1d' {category_data}")
            # concate all records to a file
            os.system(f"cat {category_data} >> {category_dir}/{filename}")
    cmder.successOut("Classify finished.")


def datatidy():
    align_filename()
    merge_data()
    classify()


def count_data_num(path: str):
    cnt = 0
    cmder.infOut("Counting records number...")
    files = list(Path(path).rglob("*.csv"))
    for file in files:
        file = str(file)
        argu = "'{print $1}'"
        _, res = cmder.runCmd(f"wc -l {file} | awk {argu}")
        res = eval(res)
        cnt += res
    cmder.successOut("Records number = " + str(cnt))


if __name__ == "__main__":
    datatidy()
