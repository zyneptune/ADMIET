import numpy as np
import os
import subprocess
from subprocess import PIPE
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# 生成数据


def func(x, p):
    # p = (a,b)
    a = p[0]
    b = p[1]
    y = a * x[:, 0] + b
    return y


def threadRun(order):
    # "../../build/functionfitting --DataX=./data/datax_1.txt --DataY=./data/datay_1.txt --ConstantUpperBound=10.0 --ConstantLowerBound=-10.0 --RandSeed=1 --NumNodes=2 --StepLength=0.05 --Epsilon=0.3 --MaxEpoch=5000 --WhenPenalty=1000"
    result = subprocess.run(
        args=order, shell=True, stdout=PIPE)
    out = result.stdout
    out = out.strip()
    out = str(out, encoding="utf-8")
    out = out.split("\n")
    out2 = np.array(out[1].split(" "))
    out2 = out2.astype(float)
    return [out[0], out2]

    # 生成自变量
if not os.path.exists("./data/datax_1.txt"):
    x = np.random.rand(100, 1) * 10.0
    np.savetxt("data/datax_1.txt", x)
if not os.path.exists("./data/parameters_1.txt"):
    p = np.random.randn(100, 2)
    np.savetxt("./data/parameters_1.txt", p)
if not os.path.exists("./data/datay_1.txt"):
    y = func(x, p[0, :])
    np.savetxt("./data/datay_1.txt", y)

x = np.loadtxt("data/datax_1.txt")
x = x.reshape([100, 1])
p = np.loadtxt("./data/parameters_1.txt")
# 生成数据并计算
path_exe = "../../build/functionfitting"
path_datax = " --DataX=./data/datax_1.txt"
# path_datay = " --DataY=./data/datay_1.txt"
bound = " --ConstantUpperBound=10.0" + " --ConstantLowerBound=-10.0"
Num_nodes = " --NumNodes=2"
Settings = " --Epsilon=0.3 --MaxEpoch=1000 --WhenPenalty=1"
result = []
for i in range(100):
    y = func(x, p[i, :])
    np.savetxt("./data/datay_1_"+str(i+1)+".txt", y)
    path_datay = " --DataY=./data/datay_1_"+str(i+1)+".txt"
    temp_results = []
    with ThreadPoolExecutor(max_workers=40) as t:
        obj_list = []
        for page in range(40):
            rs = np.random.randint(1000)
            rs = " --RandSeed=" + str(rs)
            order = path_exe + path_datax + path_datay + bound + Num_nodes + Settings + rs
            obj = t.submit(threadRun, order)
            obj_list.append(obj)
        for future in as_completed(obj_list):
            data = future.result()
            temp_results.append(data)
    # 先把所有结果保存
    temp_result2 = []
    for j in temp_results:
        with open('result/result_1_all.txt', 'a') as f:
            print(j, file=f)
        temp_result2.append(j[1])
    ttt = np.array(temp_result2)
    with_zero = np.where(ttt[:, 1] == 0)[0]
    min_row = with_zero[np.argmin(ttt[with_zero, 2])]
    result.append([ttt[min_row, 2], min_row])
    print(i+1)

re = np.array(result)
np.savetxt("result/result_1.txt", re)
