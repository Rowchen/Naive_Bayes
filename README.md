# Naive_Bayes
a simple apply for naive Bayes

代码解释：
结构体featuer_attribution:
    self.continual=continual 是否为连续型变量
    self.dic=dic             这是一个字典，如果是离散变量，则关键字是该属性的各个取值，value是一个list，存储着p(featurei=xi/c=ck)的值
                             如果是连续变量，关键字是u和sigma，value也是一个list，存储着该属性在训练集上，各个类别的均值和标准差
                             
feature_create(self,dataset,datalabel)
      在训练集上，将所有的p(featurei=xi/c=ck)都存储到对应的结构体中


                           
