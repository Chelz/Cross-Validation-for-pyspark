'''
 Created by Chellam Srinivasan
 A modular code for cross validation in pyspark
'''
def CrossValidation(k):
  countexcess={}
  countunder={}
#Split the training data into 10 folds
  trainint=(train.coalesce(10,shuffle=True).glom().collect())
  newfolds=[]
  excess=[]
  nSamplesinfold=len(train)/k
#This code snippet ensures that each fold contains equal number of examples(8000 here)
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
  for i in range(0,k):
      print ("Iteration no. {}".format(i+1))
  #Ensures that one fold is used for validation and others are used for training(without replacement i.e one fold which was earlier 
  #used for validation is never used again)   
      val=trainint[i]
      train2=trainint[:i]
      train2.extend(trainint[i+1:])
      train21=list(itertools.chain(*train2))
  #code for creation of training data and Validation data goes here
      createTrainandVal()
  #Code for Feature Extraction goes here
      featureExtract()
  #Code for Creation of Labeled Data
      createLabeledData()
  #Code for Training and Testing the model
      trainandtest()
