import pickle,joblib
with open('metaData.pkl', 'rb') as file:
    metaData = pickle.load(file)
    
with open('tree.pkl', 'rb') as file:
    tree = pickle.load(file)

with open('ward_metaData.pkl', 'rb') as file:
    ward_metaData = pickle.load(file)

with open('wards.pkl', 'rb') as file:
    wards = pickle.load(file)    
    
joblib.dump(metaData,'metaData.joblib')
joblib.dump(tree,'tree.joblib')
joblib.dump(ward_metaData,'ward_metaData.joblib')
joblib.dump(wards,'wards.joblib')