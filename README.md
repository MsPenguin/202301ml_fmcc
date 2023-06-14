# 202301ml_fmcc

train.py와 test.py 모두 실행 전에 각각 코드 내에 있는 TRAIN/TEST_DATASET_BASE의 경로 내에 데이터셋을 추가해 주어야 합니다.

train.py의 경우 실행 완료 시 "final_model.pkl" 파일을 내보내고, 이 파일이 test.py에서 사용됩니다.

test.py의 경우 실행 완료 시 "result.txt" 파일을 내보내고, 이 파일을 eval.pl에 사용합니다.