import pprint
import pickle

pkl_file=open('./data/imdb.pkl','rb')

data=pickle.load(pkl_file)
pprint.pprint(data)

pkl_file.close()

