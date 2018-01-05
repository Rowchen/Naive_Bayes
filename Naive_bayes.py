import numpy as np
import math
class featuer_attribution:
    def __init__(self,dic,continual=False):
        self.continual=continual
        self.dic=dic
class bayes:
    def __init__(self):
        self.classnum=0
        self.feature=[]
        self.per_class_num=[]
        self.classdic={}
        self.if_continual=20
    def feature_create(self,dataset,datalabel):
        self.classdic={}
        hhh = 0
        y = []
        self.per_class_num=[]
        for i in xrange(datalabel.shape[0]):
            if not self.classdic.has_key(datalabel[i]):
                self.classdic[datalabel[i]] = hhh
                hhh += 1
                self.per_class_num.append(0)
            y.append((int)(self.classdic.get(datalabel[i],0)))
            self.per_class_num[y[-1]]+=1
        self.classnum = len(self.classdic)
        feature = []
        for i in xrange(dataset.shape[1]):
            attribution_dict = {}
            flag=False
            for j in xrange(dataset.shape[0]):
                if not attribution_dict.has_key(dataset[j,i]):
                    attribution_dict[dataset[j, i]]=[0.0]*self.classnum
                attribution_dict[dataset[j, i]][y[j]]+=1
                if len(attribution_dict)>self.if_continual:#continual
                    flag=True
                    break
            if flag:
                xx=dataset[:,i]
                xxsub=[]
                for m in xrange(self.classnum):
                    xxsub.append([])
                for j in xrange(len(xx)):
                    xxsub[y[j]].append(float(xx[j]))
                mydic={'u':[],'sigma':[]}
                for j in xrange(self.classnum):
                    mydic['u'].append(np.mean(np.array(xxsub[j])))
                    mydic['sigma'].append(np.std(np.array(xxsub[j])))
                feature.append(featuer_attribution(mydic,continual=True))
            else:
                for k in attribution_dict:
                    for m in xrange(self.classnum):
                        attribution_dict[k][m]=(attribution_dict[k][m]+1)/(self.per_class_num[m]+len(attribution_dict))
                feature.append(featuer_attribution(attribution_dict))
        return feature


    def fit(self,X,y):
        self.feature=self.feature_create(X,y)
        for i in self.feature:
            print 'feature'
            print i.dic

    def predict(self,X):
        ans=[]
        for i in xrange(X.shape[0]):
            a=sum(self.per_class_num)
            multi=[0.0]*self.classnum
            for m in xrange(self.classnum):
                multi[m]=self.per_class_num[m]/float(a)
            for j in xrange(X.shape[1]):
                feature_dict=self.feature[j].dic
                if self.feature[j].continual:
                    for m in xrange(self.classnum):

                        a=math.exp(-((float(X[i, j])-feature_dict['u'][m])**2)/2/(feature_dict['sigma'][m]**2))/feature_dict['sigma'][m]/((2*3.14159)**.5)
                        multi[m]*=a
                else:
                    for m in xrange(self.classnum):
                        multi[m]*=feature_dict[X[i,j]][m]
            ans.append(multi.index(max(multi)))
        return ans


dataset = np.loadtxt("dataset\data.txt",delimiter=',',dtype=np.str)#num_train*feature_dim
X_train=dataset[:,:-1]
y=dataset[:,-1]
ss=bayes()
ss.fit(X_train,y)
print 'predict'
ans=ss.predict(X_train)
cc=0.0
yy=0.0
for i in xrange(len(ans)):
    if ans[i]==int(ss.classdic[y[i]]):
        cc+=1
    yy+=1
print cc/float(yy)

