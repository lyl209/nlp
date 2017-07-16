import json, sys
import base64
from Datum import Datum
import re

days = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
preps = ["the","at","in","of","a","an"]

class FeatureFactory:
    """
    Add any necessary initialization steps for your features here
    Using this constructor is optional. Depending on your
    features, you may not need to intialize anything.
    """
    def __init__(self):
        pass


    """
    Words is a list of the words in the entire corpus, previousLabel is the label
    for position-1 (or O if it's the start of a new sentence), and position
    is the word you are adding features for. PreviousLabel must be the
    only label that is visible to this method. 
    """

    def computeFeatures(self, words, previousLabel, position):
        features = []
        currentWord = words[position]
        pprevWord = ""
        prevWord = ""
        nextWord = ""
        if position > 0:
            prevWord = words[position-1]
        if position > 1:
            pprevWord = words[position-2]
        
        if (position+1) < len(words):
            nextWord = words[position+1]
            
        """ Baseline Features 
        precision = 0.802
        recall = 0.522
        F1 = 0.633        
        
        BEST
        precision = 0.9137026239067055
        recall = 0.855349344978166
        F1 = 0.8835635748519876

        """
        
        #features.append("word=" + currentWord)        
        #features.append("prevLabel=" + previousLabel)        
        #features.append("word=" + currentWord + ", prevLabel=" + previousLabel)        

             
        """ 
        Warning: If you encounter "line search failure" error when
        running the program, considering putting the baseline features
	back. It occurs when the features are too sparse. Once you have
        added enough features, take out the features that you don't need. 
	
         We recommend experimenting with 
         orthographic information / spelling, gazeteers / geographical , 
         and the surrounding words / ?, 
         and we also encourage you to think beyond these suggestions.

         Tips
         Generalize your features. For example, if you're adding the above "case=Title" 
         feature, 
         think about whether there is any pattern that is not captured by the feature. 
         Would the "case=Title" feature capture "O'Gorman"?
         
         When you add a new feature, think about whether it would have a positive or negative 
         weight for PERSON and O tags (these are the only tags for this assignment).
         "-print" option is useful when you want compare the differences of the predictions before and after adding new features.

        """
                
        l = len(currentWord)
    
        # 1 'word' - normalize all upper case words, learn the word for PERSON
       
        if currentWord.isupper():
            features.append("word=" + currentWord.title())
        else:
            features.append("word=" + currentWord)        
        
        
        # high recall
        # 2 'newcap' - identify this is a new capital (potentially a PERSON)                      
        if l > 2 and currentWord.istitle():
            features.append("newcap=" +  currentWord.title())
        #else:
        #    features.append("word=" + currentWord)   
            
        # 3 'prep' - previous word is prepositions, add positive weight to O
        
        if prevWord.isdigit():
            features.append("prev=d")
        elif prevWord.lower() in preps:
            features.append("prev=" + prevWord.lower())
        elif pprevWord.lower() in preps:
            features.append("prev=" + pprevWord.lower())
        elif nextWord.lower() in preps:
            features.append("next=" + nextWord.lower())
                
        # high precision
        # 4 'follower' - identify PERSON after PERSON, positive weight to PERSON
        if previousLabel == "PERSON" and l >= 2 and currentWord[0].isupper():
            features.append("follower=y")
#            if currentWord[1].islower() or currentWord[1] == "'" or currentWord[1] == ".":
#                features.append("follower=y")
#            else:
#                # mostly all caps
#                if nextWord == "says" or nextWord == "said" or nextWord == "told" or nextWord == "'S" or nextWord == "'s":
#                    features.append("follower=y")


        # 5 'lower' - lower case, adds positive weights to O
        
        if not currentWord[0].isupper():
            features.append("lower=y")

        # 6 'next' - next word

        if nextWord == "said" or nextWord == "'s" or nextWord == "'S" or nextWord == "told" or nextWord == "says" or nextWord == "who" or nextWord == "whose":
            features.append("next=" + nextWord)

        #if currentWord == "discard" and nextWord == "Andy":
        #    print features
        
        return features

    """ Do not modify this method """
    def readData(self, filename):
        data = [] 
        
        for line in open(filename, 'r'):
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data

    """ Do not modify this method """
    def readTestData(self, ch_aux):
        data = [] 
        
        for line in ch_aux.splitlines():
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data


    """ Do not modify this method """
    def setFeaturesTrain(self, data):
        newData = []
        words = []

        for datum in data:
            words.append(datum.word)

        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        previousLabel = "O"
        for i in range(0, len(data)):
            datum = data[i]

            newDatum = Datum(datum.word, datum.label)
            newDatum.features = self.computeFeatures(words, previousLabel, i)
            newDatum.previousLabel = previousLabel
            newData.append(newDatum)

            previousLabel = datum.label

        return newData

    """
    Compute the features for all possible previous labels
    for Viterbi algorithm. Do not modify this method
    """
    def setFeaturesTest(self, data):
        newData = []
        words = []
        labels = []
        labelIndex = {}

        for datum in data:
            words.append(datum.word)
            if not labelIndex.has_key(datum.label):
                labelIndex[datum.label] = len(labels)
                labels.append(datum.label)
        
        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        for i in range(0, len(data)):
            datum = data[i]

            if i == 0:
                previousLabel = "O"
                datum.features = self.computeFeatures(words, previousLabel, i)

                newDatum = Datum(datum.word, datum.label)
                newDatum.features = self.computeFeatures(words, previousLabel, i)
                newDatum.previousLabel = previousLabel
                newData.append(newDatum)
            else:
                for previousLabel in labels:
                    datum.features = self.computeFeatures(words, previousLabel, i)

                    newDatum = Datum(datum.word, datum.label)
                    newDatum.features = self.computeFeatures(words, previousLabel, i)
                    newDatum.previousLabel = previousLabel
                    newData.append(newDatum)

        return newData

    """
    write words, labels, and features into a json file
    Do not modify this method
    """
    def writeData(self, data, filename):
        outFile = open(filename + '.json', 'w')
        for i in range(0, len(data)):
            datum = data[i]
            jsonObj = {}
            jsonObj['_label'] = datum.label
            jsonObj['_word']= base64.b64encode(datum.word)
            jsonObj['_prevLabel'] = datum.previousLabel

            featureObj = {}
            features = datum.features
            for j in range(0, len(features)):
                feature = features[j]
                featureObj['_'+feature] = feature
            jsonObj['_features'] = featureObj
            
            outFile.write(json.dumps(jsonObj) + '\n')
            
        outFile.close()

