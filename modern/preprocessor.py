import re
import pandas as pd
import os
import cmder

from pathlib import Path

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
        if not os.path.isdir(new_dir):
            os.system(f"mv {dir} {new_dir}")
        else:
            os.system(f"mv {dir}/* {new_dir}")
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
            new_data = category_data[len(DATA) :].replace("/", "_")
            os.system(f"cp {category_data} {category_dir}/{new_data}")
    cmder.successOut("Classify finished.")


def datatidy():
    align_filename()
    merge_data()
    classify()


def count_data_num():
    cnt = 0
    cmder.infOut("Counting records number...")
    files = list(Path(DATA).rglob("video_*.csv"))
    for file in files:
        file = str(file)
        argu = "'{print $1}'"
        _, res = cmder.runCmd(f"wc -l {file} | awk {argu}")
        res = eval(res)
        cnt += res
    cmder.successOut("Records number = " + str(cnt))


class Set:
    def __init__(self, cls: str):
        self.cls = cls
        self.hw = HW
        self.pw = PW
        self.fps = FPS
        self.downsample = DOWNSAMPLE

    def set_index_read(self, file=SPLIT_PATH):
        file = os.path.abspath(file)
        data = pd.read_excel(
            file, sheet_name=0 if self.cls == "train" else 1, names=["index"]
        )
        return data.values.flatten().tolist()

    def single_video_process(self, video_name):
        path = video_name
        # 2D list
        data = []
        video_name = video_name.split("/")
        video_name = int(video_name[-1][:3])
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "")
                line = line.split(",")
                # Just append head position x,y and gaze position x,y as a list
                # [head_x, head_y]
                data.append([line[3], line[4]])
        fin_data = []
        for i in range(0, len(data), self.fps):
            k = []
            hw_end = i + self.hw * self.fps
            pw_end = hw_end + self.pw * self.fps
            if pw_end >= len(data):
                break
            # sample history window data
            for m in range(i, hw_end, self.downsample):
                for n in range(2):
                    k.append(data[m][n])
            # sample predict window data
            for m in range(hw_end, pw_end, self.downsample):
                for n in range(2):
                    k.append(data[m][n])
            fin_data.append(k)
        return fin_data

    def create_set(self):
        idx_files = self.set_index_read(SPLIT_PATH)
        users = os.listdir(SET_PATH + "Gaze_txt_files/")
        set = []
        for user in users:
            for video in os.listdir(SET_PATH + "Gaze_txt_files/" + user + "/"):
                video_path = SET_PATH + "Gaze_txt_files/" + user + "/" + video
                # extract data for train or test seperately:
                if int(video[:3]) in idx_files:
                    data = self.single_video_process(video_path)
                    set += data
        return set


def create_dataset(cls: str):
    set = Set(cls).create_set()
    df_set = pd.DataFrame(set)
    print(df_set)
    print(df_set.shape)
    csv_name = MODELS_PATH + cls + ".csv"
    df_set.to_csv(csv_name, header=False, index=None, mode="w")


if __name__ == "__main__":
    datatidy()
