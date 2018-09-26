import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

name = "Cassidy"

path = "C:\\Users\\Cass\\Desktop\\names"
files = glob.glob(os.path.join(path, "*.txt"))

male = []
female = [] 
year = [] 

for file in files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    year.append(int(file_name[3:]))
    file = pd.read_csv(file, header = None)
    male_raw = str(file.loc[(file[0] == name) & (file[1] == "M"), 2].values)
    if male_raw == "[]":
        m_occur = 0
    else:
        m_occur = int(male_raw[1:-1])
    male.append(m_occur)
    female_raw = str(file.loc[(file[0] == name) & (file[1] == "F"), 2].values)
    if female_raw == "[]":
        f_occur = 0
    else:
        f_occur = int(female_raw[1:-1])
    female.append(f_occur) 

m_to_f_ratio = []    
    
for i, j in zip(male, female):
    if i !=0 and j!=0:
        m_to_f_ratio.append(i/j)
    else: m_to_f_ratio.append(0)
    
plt.plot(year, male, label="male")
plt.plot(year, female, label="female")
plt.legend(loc='upper left')
plt.title("Births per year in US by name: {}".format(name))
plt.grid()
plt.locator_params(numticks=12)
plt.show()

plt.plot(year, m_to_f_ratio)
plt.title("Ratio of males to females per year for name: {}".format(name))
plt.grid()
plt.locator_params(numticks=12)
plt.show()
