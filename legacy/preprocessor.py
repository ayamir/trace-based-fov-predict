import pandas as pd
import os
import cmder

SPLIT_PATH = "../../dataset/shanghai_gaze/train_test_set.xlsx"
SET_PATH = "../../dataset/shanghai_gaze/"

HW = 1
PW = 1
FPS = 30
DOWNSAMPLE = 2
MODELS_PATH = "../../models/legacy/hw" + str(HW) + "pw" + str(PW) + "models/"


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
    if not os.path.isdir(MODELS_PATH):
        cmder.runCmd(f"mkdir -p {MODELS_PATH}")
    create_dataset("train")
    create_dataset("test")
