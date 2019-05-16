from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import pickle
import linecache

read_file='E:/data_out1.txt'

#with open('E:/test.txt','r',encoding='utf8') as f:
#	document = f.readlines() #读取所有行并返回列表
document =linecache.getlines(read_file)
    
        
    

vectorizer = CountVectorizer(decode_error="replace")
tfidftransformer = TfidfTransformer()
# 注意在训练的时候必须用vectorizer.fit_transform、tfidftransformer.fit_transform
# 在预测的时候必须用vectorizer.transform、tfidftransformer.transform
vec_train = vectorizer.fit_transform(document)
tfidf = tfidftransformer.fit_transform(vec_train)

# 保存经过fit的vectorizer 与 经过fit的tfidftransformer,预测时使用
feature_path = 'models/feature.pkl'
with open(feature_path, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)

tfidftransformer_path = 'models/tfidftransformer.pkl'
with open(tfidftransformer_path, 'wb') as fw:
    pickle.dump(tfidftransformer, fw)
    
linecache.clearcache()