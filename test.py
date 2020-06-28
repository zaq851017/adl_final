import os
import sys
print("Preprocessing...")
os.system("python3.6 preprocess.py --test_path " + sys.argv[1])
print("Predicting...")
os.system("python3.6 predict.py --model model/bert.pth --test_path " + sys.argv[1] +" --mode test --backbone bert")