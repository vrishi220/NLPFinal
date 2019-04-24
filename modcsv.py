import pandas as pd
import glob, random

path = './#stuff/*.txt'
files = glob.glob(path)
fp,sd,sdict,used,fint,finl = {},{},{},[],[],[]

fnames = ["#anxiety","#depression","#suicide","#neutral"]

df = pd.read_csv("info.csv",encoding="latin1")


for each in fnames: sdict[each] = df[df['CATEGORY'] == each]


for f in files: fp[f] = [e for e in open(f,'r')]


for each in sdict: sd[each] = (list(sdict[each].TITLE))


for each in sd:
	if each == '#neutral': continue
	for e in fp:
		if e.endswith(each+'.txt'):
			for e2 in fp[e]:
				for _ in range(40):
					t = random.randint(0, len(sd[each]) - 1)
					while t in used: t = random.randint(0, len(sd[each]) - 1)
					sd[each][t] = e2
					used.append(t)


for each in sd:
	for e in sd[each]: fint.append(e);finl.append(each)
        

sdfin = pd.DataFrame(fint,finl)
sdfin.to_csv(r"u2_info.csv",index=True)
