cd /data2/monodetr-0722/train_1
nohup python tools/train_val.py --config configs/monodetr.yaml >logs/0629_1_train.log 2>&1 &
nohup python tools/train_val.py --config configs/monodetr_e.yaml >logs/0801_1_test.log -e 2>&1 &