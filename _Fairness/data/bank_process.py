import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
job = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"]
marital = ["married","divorced","single"]
education = ["unknown","secondary","primary","tertiary"]
default = ["no","yes"]
housing = ["no","yes"]
loan = ["no","yes"]
contact = ["unknown","telephone","cellular"]
month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
poutcome = ["unknown","other","failure","success"]
output = ["no","yes"]

data = []
with open("../datasets/bank_raw/bank-full.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(';')
        features[0] = np.clip(int(features[0]) / 10, 1, 9)
        features[1] = job.index(features[1].split('\"')[1])
        features[2] = marital.index(features[2].split('\"')[1])
        features[3] = education.index(features[3].split('\"')[1])
        features[4] = default.index(features[4].split('\"')[1])
        features[5] = np.clip(int(features[5]) / 50, -20, 180 - 1)
        features[6] = housing.index(features[6].split('\"')[1])
        features[7] = loan.index(features[7].split('\"')[1])
        features[8] = contact.index(features[8].split('\"')[1])
        features[9] = int(features[9])
        features[10] = month.index(features[10].split('\"')[1])
        features[11] = np.clip(int(features[11]) / 10, 0, 100 - 1)
        features[12] = int(features[12])
        if int(features[13])==-1:
            features[13] = 1
        else:
            features[13] = 0
        features[14] = np.clip(int(features[14]), 0, 1)
        features[15] = poutcome.index(features[15].split('\"')[1])
        features[16] = output.index(features[16].split('\"')[1])
        data.append(features)
data = np.asarray(data)

np.savetxt("../datasets/bank", data, fmt="%d",delimiter=",")