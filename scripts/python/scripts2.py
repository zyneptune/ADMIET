import numpy as np
import os
import subprocess
from subprocess import PIPE
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# 生成数据


def func(x, p):
    # p = (a,b,c)
    a = p[0]
    b = p[1]
    c = p[2]
    y = a * x[:, 0] * x[:, 0] + b * x[:, 0] + c
    return y


def threadRun(order):
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
if not os.path.exists("./data/datax_2.txt"):
    x = np.random.randn(100, 1) * 10.0
    np.savetxt("data/datax_2.txt", x)

if not os.path.exists("./data/parameters_2.txt"):
    p = np.random.randn(100, 3)
    np.savetxt("./data/parameters_2.txt", p)

x = np.loadtxt("data/datax_2.txt")
x = x.reshape([100, 1])
p = np.loadtxt("./data/parameters_2.txt")

# 生成数据并计算
path_exe = "../../build/functionfitting"
path_datax = " --DataX=./data/datax_2.txt"
# path_datay = " --DataY=./data/datay_1.txt"
bound = " --ConstantUpperBound=10.0" + " --ConstantLowerBound=-10.0"
Num_nodes = " --NumNodes=5"
Settings = " --Epsilon=0.5 --MaxEpoch=2000 --WhenPenalty=500"
result = []
for i in range(100):
    y = func(x, p[i, :])
    np.savetxt("./data/datay_2_"+str(i+1)+".txt", y)
    path_datay = " --DataY=./data/datay_2_"+str(i+1)+".txt"
    temp_results = []
    with ThreadPoolExecutor(max_workers=40) as t:
        obj_list = []
        for page in range(40):
            rs = np.random.randint(10000)
            rs = " --RandSeed=" + str(rs)
            order = path_exe + path_datax + path_datay + bound + Num_nodes + Settings + rs
            obj = t.submit(threadRun, order)
            obj_list.append(obj)
        for future in as_completed(obj_list):
            data = future.result()
            temp_results.append(data)
    temp_result2 = []
    for j in temp_results:
        with open('result/result_2_all.txt', 'a') as f:
            print(j, file=f)
        temp_result2.append(j[1])
    ttt = np.array(temp_result2)
    with_zero = np.where(ttt[:, 1] == 0)[0]
    min_row = with_zero[np.argmin(ttt[with_zero, 2])]
    result.append([ttt[min_row, 2], min_row])
    print(i+1)

re = np.array(result)
np.savetxt("result/result_2.txt", re)
