import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#import japanize_matplotlib
import os
import argparse
from sklearn.metrics import r2_score
import scipy.optimize as optimize
import sympy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments （help是该参数的提示信息）
    parser.add_argument("-s", action="store", dest='std_data', required=True, help="std_data")
    parser.add_argument("-e", action="store", dest='elisa_data', required=True, help="elisa_data")
    parser.add_argument("-d", action="store", dest='dilution_ratio', required=True, help="dilution_ratio")
    parser.add_argument("-od", action="store", dest='out_dir', required=True, help="out_dir")

    # Parse the command-line arguments （获得传入的参数）
    args = parser.parse_args()

    # Access the values of the arguments
    std_data= args.std_data
    elisa_data = args.elisa_data
    dilution_ratio = args.dilution_ratio
    out_dir = args.out_dir

    path = out_dir
    # mkdir
    if not os.path.exists(path):
        os.mkdir(path)

#if __name__ == '__main__':
#   print ('test')

std= pd.read_excel(std_data, index_col=0, header=0)

x_std=std.columns.values
y_std=std.values[0]

##cancel RuntimeWarning: divide by zero encountered
np.seterr(divide='ignore', invalid='ignore')
#近似させる関数の定義
def func(x, a, b, c,d ):
    return   d + (a - d) / (1 + (x / c) ** b)

# 近似関数を求める
#optimize.curve_fit return to -> truple (popt,pcov)
popt, _ = optimize.curve_fit(func, x_std, y_std)
# 近似関数の決定係数(R2)を求める
#r2 = metrics.r2_score(y_std, func(x_std, *popt))

plt.figure(figsize=(12,7))
plt.errorbar(x_std, y_std, capsize=5, fmt='o', markersize=4, ecolor='black', markeredgecolor = "black", color='w')
plt.plot(x_std, func(x_std, *popt),  color = 'black', linestyle=':')
r2 = r2_score(y_std, func(x_std, *popt))
plt.text(max(x_std)/2.5, max(y_std)*0.9,
         'y = {:.4f}+({:.4f}-{:.4f})/(1 + (x /{:.4f})^{:.4f}))\n$R^2$ = {:.4f}'.format(popt[3],popt[0],popt[3],popt[2],popt[1],r2),
         fontsize=10) #max(x)/3.7和max(y)*6/8是文本标签的位置，分别代表x轴和y轴上的坐标
plt.xlabel(std.index.name, fontsize=12)
plt.ylabel(std.index[0], fontsize=12)
plt.savefig(os.path.join(path, '標準品検量線.png'))

elisa= pd.read_excel(elisa_data, index_col=0, header=0)
y_elisa=elisa.apply(lambda x:x.mean(),axis=0).values
y_elisa_err=elisa.apply(lambda x:x.std(),axis=0).values  ##data.std(ddof=1) #1是默认值
x_elisa=elisa.columns.values

plt.figure(figsize=(12,7))
plt.bar(range(len(x_elisa)),y_elisa,tick_label=x_elisa,yerr=y_elisa_err, capsize=4,color='C1')  #第一个参数为x轴元素的个数， 第二个参数为y轴的值，tick_label为x轴各元素的值
plt.ylabel(elisa.index[0],loc='top',rotation='horizontal',labelpad=-20)
plt.xlabel(elisa.index.name,
           rotation=45, #标签方向
           labelpad=-30, #调整x轴标签与x轴距离
           x=-0.06)  #调整x轴标签的左右位置
plt.xticks(rotation=45) #刻度方向
plt.savefig(os.path.join(path, 'ELISA吸光度.png'))

popt_=np.round(popt,4)
a,b,c,d=popt_

def defunc(Y):
    X = sympy.symbols("X") # 申明未知数"x"
    ls = sympy.solve([d + (a - d) / (1 + (X / c)**b)-Y],[X]) # 写入需要解的方程体
    ls_solution1=ls[0]
    ls_solution=ls_solution1[0]
    return ls_solution

concn_d = elisa.applymap(defunc)

with open(dilution_ratio,"r") as file:
    d_r = float(file.read())
    print(d_r)

concn = concn_d.applymap(lambda x: x * d_r)
concn.to_csv(os.path.join(path,'検体濃度.csv'))