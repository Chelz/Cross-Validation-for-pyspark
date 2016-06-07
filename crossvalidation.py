def CrossValidation(k):
  countexcess={}
  countunder={}
  #Split the training data into k folds(train is the complete training data)
  trainint=(train.coalesce(k,shuffle=True).glom().collect())
  newfolds=[]
  excess=[]
  nSamplesinfold=len(train)/k
#This code snippet ensures that each fold contains equal number of examples(nSamplesinfold here)
  for i,val in enumerate(trainint):
      if len(val)-nSamplesinfold>0:
          countexcess[i]=len(val)-nSamplesinfold
          trainint[i]=val[countexcess[i]:]
          excess.append(val[:countexcess[i]+1])
      else:
          countunder[i]=nSamplesinfold-len(val)
  excess=list(itertools.chain(*excess))
  for key,value in countunder.items():
      ex=excess[:value]
      trainint[key].extend(ex)
  accuracy_nb_val=[]
  accuracy_nb_test=[]
  for i in range(0,k):
      print ("Iteration no. {}".format(i+1))
#Ensures that one fold is used for validation and others are used for training(without replacement i.e one fold which was earlier 
#used for validation is never used again)   
      val=trainint[i]
      train2=trainint[:i]
      train2.extend(trainint[i+1:])
      train21=list(itertools.chain(*train2))
#form the training and validation data
      train=sc.parallelize(train21)
      validation=sc.parallelize(val)
      hashingTF = HashingTF()
#Feature Extraction for training data
      tf_train = train.map(lambda tup: hashingTF.transform(tup[1]))
      idf_train = IDF().fit(tf_train)
      tfidf_train = idf_train.transform(tf_train)
#feature Extraction for Validation Data
      tf_val = validation.map(lambda tup: hashingTF.transform(tup[1]))
      idf_val = IDF().fit(tf_val)
      tfidf_val = idf_train.transform(tf_val)
#Feature Extraction for test data
      tf_test = test.map(lambda tup: hashingTF.transform(tup[1]))
      idf_test = IDF().fit(tf_test)
      tfidf_test = idf_test.transform(tf_test)
#Labeled Data for training data
      labels=train.map(lambda x:x[0])
      transformeddata=labels.zip(tfidf_train)
      labeled1_train = transformeddata.map(lambda k: LabeledPoint(k[0], k[1]))
#Labeled Data for validation data
      labelsval=validation.map(lambda x:x[0])
      transformedvaldata=labelsval.zip(tfidf_val)
      labeled1_val = transformedvaldata.map(lambda k: LabeledPoint(k[0], k[1]))
#Labeled Data for Test Data
      labelstest=test.map(lambda x:x[0])
      transformeddatatest=labelstest.zip(tfidf_test)
      labeled1_test = transformeddatatest.map(lambda k: LabeledPoint(k[0], k[1]))
      model = NaiveBayes.train(labeled1_train, 1.0)
      predictionAndLabelval = labeled1_val.map(lambda p: (model.predict(p.features), p.label))
      accuracyval = 1.0 * predictionAndLabelval.filter(lambda x: x[0] == x[1]).count() / validation.count()
      print('model Validation accuracy for Naive Bayes{}'.format(accuracyval))
      accuracy_nb_val.append(accuracyval)
      predictionAndLabeltest = labeled1_test.map(lambda p: (model.predict(p.features), p.label))
      accuracytest = 1.0 * predictionAndLabeltest.filter(lambda x: x[0] == x[1]).count() / test.count()
      print('model test accuracy for Naive Bayes{}'.format(accuracytest))
      accuracy_nb_test.append(accuracytest)


