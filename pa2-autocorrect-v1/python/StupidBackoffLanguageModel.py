import math
class StupidBackoffLanguageModel:

    #use an unsmoothed bigram model combined with backoff to an add-one smoothed unigram model
    #Stupid Backoff Language Model: 0.18
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.uniwords = {}
    self.biwords = {}    
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        for i in range(len(sentence.data)):
            self.total = self.total +1
            unitok = sentence.data[i].word
            if unitok in self.uniwords:
                self.uniwords[unitok] = self.uniwords[unitok] + 1
            else:
                self.uniwords[unitok] = 1
            
            if (i+1) < len(sentence.data):
                bitok = unitok + ' ' + sentence.data[i+1].word
                if bitok in self.biwords:
                    self.biwords[bitok] = self.biwords[bitok] + 1
                else:
                    self.biwords[bitok] = 1    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """    
    score = 0.0 
    
    for i in range(1, len(sentence)):
        countw1 = 0.0
        countw2 = 0.0
        bicount = 0.0
        
        if sentence[i-1] in self.uniwords:
            countw1 = self.uniwords[sentence[i-1]]

        if sentence[i] in self.uniwords:
            countw2 = self.uniwords[sentence[i]]
        
        bitok = sentence[i-1] + ' ' + sentence[i]
        if bitok in self.biwords:
            bicount = self.biwords[bitok]
        
        if bicount > 0:
            score += math.log( float(bicount) / countw1 )
        else:
            score += math.log(0.4) + math.log(float(countw2+1) / (len(self.uniwords.keys())+self.total) )
    
    #print "returning", score
    return score
