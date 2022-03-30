import re
import os
import cmder
import random

from pathlib import Path
import pandas as pd

DATA = "../../dataset/xr_data/"
DATA1 = "../../dataset/xr_data/Experiment_1/"
DATA2 = "../../dataset/xr_data/Experiment_2/"

HW = 1
PW = 1
FPS = 30
DOWNSAMPLE = 2
MODELS_PATH = "../../models/modern/hw" + str(HW) + "pw" + str(PW) + "/"


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
    cmder.successOut("Aligning completed.")


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
    cmder.successOut("Merge completed.")


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
    cmder.successOut("Classify completed.")


def datatidy():
    align_filename()
    merge_data()
    classify()


def to_dataset(source_path: str, dest_path: str):
    data = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace("\n", "")
            line = line.split(",")
            # [orientation.x, orientation.y, orientation.z, orientation.w]
            data.append([line[2], line[3], line[4], line[5]])
    set_data = []
    for i in range(0, len(data), FPS):
        k = []
        hw_end = i + HW * FPS
        pw_end = hw_end + PW * FPS
        if pw_end >= len(data):
            break
        # sample history window data
        for m in range(i, hw_end, DOWNSAMPLE):
            for n in range(4):
                k.append(data[m][n])
        # sample predict window data
        for m in range(hw_end, pw_end, DOWNSAMPLE):
            for n in range(4):
                k.append(data[m][n])
        set_data.append(k)
    df_set = pd.DataFrame(set_data)
    print(df_set)
    print(df_set.shape)
    df_set.to_csv(dest_path, header=False, index=None, mode="w")


def construct(isCategory: bool):
    cwd = os.path.dirname(os.path.realpath(__file__))
    if not isCategory:
        cmder.infOut("Constructing uncategorized dataset...")
        all_dir = DATA + "All"
        os.system(f"mkdir -p {all_dir} {MODELS_PATH}")
        train_temp = cwd + "/train_temp.csv"
        test_temp = cwd + "/test_temp.csv"
        train_file = MODELS_PATH + "train.csv"
        test_file = MODELS_PATH + "/test.csv"
        if os.path.isfile(train_temp):
            os.remove(train_temp)
        if os.path.isfile(test_temp):
            os.remove(test_temp)
        for category in category_dict:
            os.system(f"cp {DATA}/{category}/* {all_dir}")
        filepaths = list(Path(all_dir).rglob("video*.csv"))
        files = []
        for file in filepaths:
            file = str(file)
            files.append(file)
        random.shuffle(files)
        # train.csv
        cmder.infOut("Constructing train.csv")
        trains = files[:15]
        for train in trains:
            os.system(f"cat {train} >> {train_temp}")
        to_dataset(train_temp, train_file)
        # test.csv
        cmder.infOut("Construction test.csv")
        tests = files[15:]
        for test in tests:
            os.system(f"cat {test} >> {test_temp}")
        to_dataset(test_temp, test_file)
        os.system(f"rm {train_temp} {test_temp}")
    cmder.successOut("Construction completed.")


if __name__ == "__main__":
    construct(False)
