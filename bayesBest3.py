# Name: Fan Wu
# Date: 5.11.2015
# Description:
#
#

import math, os, pickle, re
import nltk
from nltk.util import ngrams
class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""

      #use flag to tell if the text is positive or negative
      self.flag = 0
      
      try:
         self.positive = self.load("store_positive_Best")
         self.negative = self.load("store_negative_Best")
         self.positive_number = self.load("store_positive_number_Best")
         self.negative_number = self.load("store_negative_number_Best")
      except IOError:
         self.positive = {}
         self.negative = {}
         self.positive_number = 0
         self.negative_number = 0
         self.train()
   def eval_test(self):
         iFileList = []
         classify_positive = 0
         correct = 0
         for fFileObj in os.walk("movies_reviews/"):
            iFileList = fFileObj[2]
            break
         for i in iFileList:	
            txt = self.loadFile(i)
            result = self.classify(txt)
            print result
            if result == "positive":
               classify_positive += 1
               if i.find("-5-") != -1:
                  correct += 1

         precision = float(correct) / float(classify_positive)
         recall = float(correct) / float(self.positive_number)
         fmeasure = 2 * float(precision) * float(recall) / (float(precision) + float(recall))
         print correct 
         print classify_positive
         print self.positive_number
         print precision 
         print recall
         print fmeasure


   def eval(self):
     
         iFileList = []
         positiveFileList = []
         negativeFileList = []
         correct_negative = 0
         lassify_negative = 0


         for fFileObj in os.walk("movies_reviews/"):
            iFileList = fFileObj[2]
            break

         # separate iFileList into positiveFileList which contains only positve files' names
         # and negativeFileList which contains only negative files' names
         for i in iFileList:
            if i.find("-5-") != -1:
               positiveFileList.append(i)
            else:
               negativeFileList.append(i)

         positive_10fold_number = len(positiveFileList) / 10 + 1 if len(positiveFileList) % 10 != 0 else len(positiveFileList) / 10
         negative_10fold_number = len(negativeFileList) / 10 + 1 if len(negativeFileList) % 10 != 0 else len(negativeFileList) / 10

         train_set = [0]*10
         test_set = [0]*10
         precision_result = 0.0
         recall_result = 0.0
         fmeasure_result = 0.0
         precision_result_negative = 0.0
         recall_result_negative = 0.0
         fmeasure_result_negative = 0.0


         for i in range(10):
            if i != 9:
               test_set[i] = positiveFileList[i*positive_10fold_number:(i+1)*positive_10fold_number] + negativeFileList[i*negative_10fold_number:(i+1)*negative_10fold_number]
            else:
               test_set[i] = positiveFileList[i*positive_10fold_number:] + negativeFileList[i*negative_10fold_number:]

            train_set[i] = list(set(iFileList) - set(test_set[i]))

w

               positive_dict = self.load(positive_filename)
               negative_dict = self.load(negative_filename)
            except IOError:
               self.trainForEval(train_set[i], i)
               positive_dict = self.load(positive_filename)
               negative_dict = self.load(negative_filename)


            for k in test_set[i]:
               txt = self.loadFile(k)
               result = self.classifyForEval(txt,positive_dict,negative_dict)
               if result == "positive":
                  classify_positive += 1
                  if k.find("-5-") != -1:
                     correct +=1
	       if result == "negative":
	          classify_negative += 1
		  if k.find("-1-") != -1:
		     correct_negative += 1

            precision= float(correct) / float(classify_positive)
            recall = float(correct) / float(self.positive_number * 0.1)
            fmeasure = 2 * float(precision) * float(recall) / (float(precision) + float(recall))
	    precision_result += precision
	    recall_result += recall
	    fmeasure_result += fmeasure

	    precision_negative= float(correct_negative) / float(classify_negative)
            recall_negative = float(correct_negative) / float(self.negative_number*0.1)
            fmeasure_negative = 2 * float(precision_negative) * float(recall_negative) / (float(precision_negative) + float(recall_negative))
	    precision_result_negative += precision_negative
	    recall_result_negative += recall_negative
	    fmeasure_result_negative += fmeasure_negative
	    print correct 
            print classify_positive
            print self.positive_number*0.1




         print precision_result / 10.0 
         print recall_result / 10.0
         print fmeasure_result / 10.0
	 print precision_result_negative / 10.0 
         print recall_result_negative / 10.0
         print fmeasure_result_negative / 10.0


   def trainForEval(self, train_set, i):
      """
      train the native bayes sentiment for evaluation
      """
      negative_dict = {}
      positive_dict = {}
      negative_filename = "negative_" + str(i) + "_Best"
      positive_filename = "positive_" + str(i) + "_Best"

      flag = 0
      for i in train_set:
         flag = 0
         if i.find("-5-") != -1:
            flag = 1
         else:
            flag = 2

         file_text = self.loadFile(i)
         file_list = nltk.word_tokenize(file_text.decode('utf-8'))
         file_list = ngrams(file_list,2)
	 file_list = list(set(file_list))

         for j in file_list:
            if flag == 2:
               if j not in negative_dict.keys():
                  negative_dict[j] = 1
               else:
                  negative_dict[j] += 1
            else:
               if j not in positive_dict.keys():
                  positive_dict[j] = 1
               else:
                  positive_dict[j] +=1

      self.save(positive_dict, positive_filename)
      self.save(negative_dict, negative_filename)

      return 


   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      iFileList = []
      for fFileObj in os.walk("movies_reviews/"):
         iFileList=fFileObj[2]
         break
      for i in iFileList:
         self.flag = 0
         if i.find("-5-") != -1 :
            self.flag = 1
            self.positive_number +=1
         else:
            self.flag = 2
            self.negative_number +=1

         file_text = self.loadFile(i)
         file_list = nltk.word_tokenize(file_text.decode('utf-8'))
	 
         file_list = ngrams(file_list,2) 
	 #file_list is the presence of the word in the text, if a word appears twice, only counts one.
         file_list = list(set(file_list))

         for j in file_list:
            if self.flag == 1:
               if j not in self.positive.keys():
                  self.positive[j] = 1
               else:
                  self.positive[j] += 1
            else:
               if j not in self.negative.keys():
                  self.negative[j] = 1
               else:
                  self.negative[j] += 1
	 print file_list
      self.save(self.positive, "store_positive_Best")
      self.save(self.negative, "store_negative_Best")
      self.save(self.positive_number, "store_positive_number_Best")
      self.save(self.negative_number, "store_negative_number_Best")


   def classifyForEval(self, sText, positive_dict, negative_dict):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral),
      according to the positive_dict and negative_dict Given.
      """

      #we assume number of positive text equals number of negative text. so we eliminate the
      #calculation of the number of positive / (the number of all text)
      lTokens = nltk.word_tokenize(sText.decode('utf-8'))
      lTokens = ngrams(lTokens,2) 
      P_positive_result = 0
      P_negative_result = 0

      for i in lTokens:
         #print P_positive_result, P_negative_result
         if positive_dict.has_key(i) == False and negative_dict.has_key(i) == False:
            continue
         elif positive_dict.has_key(i) == True and negative_dict.has_key(i) == False:
            P_positive_result += math.log(float(positive_dict[i]+1)/(float(self.positive_number)*0.9))
            P_negative_result += math.log(1.0/(float(self.negative_number)*0.9))
         elif positive_dict.has_key(i) == False and negative_dict.has_key(i) == True:
            P_positive_result += math.log(1.0/(float(self.positive_number)*0.9))
            P_negative_result += math.log(float(negative_dict[i]+1)/(float(self.negative_number)*0.9))

         else: 
            P_positive_result += math.log(float(positive_dict[i])/(float(self.positive_number)*0.9))
            P_negative_result += math.log(float(negative_dict[i])/(float(self.negative_number)*0.9))

      if P_positive_result > P_negative_result :
         return "positive"
      elif P_positive_result < P_negative_result:
         return "negative"
      else:
         return "neutral" 
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      #we assume number of positive text equals number of negative text. so we eliminate the
      #calculation of the number of positive / (the number of all text)
      lTokens = nltk.word_tokenize(sText.decode('utf-8'))
      lTokens = ngrams(lTokens,2) 
      P_positive_result = 0
      P_negative_result = 0

      for i in lTokens:
         #print P_positive_result, P_negative_result
         if self.positive.has_key(i) == False and self.negative.has_key(i) == False:
            continue
         elif self.positive.has_key(i) == True and self.negative.has_key(i) == False:
            P_positive_result += math.log(float(self.positive[i]+1)/float(self.positive_number))
            P_negative_result += math.log(1.0/float(self.negative_number))
         elif self.positive.has_key(i) == False and self.negative.has_key(i) == True:
            P_positive_result += math.log(1.0/float(self.positive_number))
            P_negative_result += math.log(float(self.negative[i]+1)/float(self.negative_number))

         else: 
            P_positive_result += math.log(float(self.positive[i])/float(self.positive_number))
            P_negative_result += math.log(float(self.negative[i])/float(self.negative_number))

      if P_positive_result > P_negative_result :
         return "positive"
      elif P_positive_result < P_negative_result:
         return "negative"
      else:
         return "neutral" 



   def calculate(self, sText, flag):
      """Given a target string sText, this function returns the possibility the document 
      belongs to the certain document class. flag = 1 means positive, flag = 2 means negative
      """


   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""
      sFilename = "movies_reviews/" + sFilename
      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens
