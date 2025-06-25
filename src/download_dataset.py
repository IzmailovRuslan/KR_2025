import os

os.system("wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip -d refcoco/ && rm train2014.zip")
os.system("wget wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip && unzip refcoco.zip -d annotations/ && rm refcoco.zip")