from utils import load_dataset


TRAIN_PATH_LIST = "fmcc_train.ctl"
TEST_PATH_LIST = "fmcc_test.ctl"

TRAIN_DATASET_BASE = "raw16k/train/"
TEST_DATASET_BASE = "raw16k/test/"

audio_male = []  # 남성 음성 데이터
audio_female = []  # 여성 음성 데이터

audio_male, audio_female = load_dataset(TRAIN_DATASET_BASE)

print(audio_male)