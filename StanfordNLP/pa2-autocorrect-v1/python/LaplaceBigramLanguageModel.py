import math
class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.uniwords = {}
    self.biwords = {}    
        
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        for i in range(len(sentence.data)):
                        
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
        unicount = 0.0
        bicount = 0.0
        
        if sentence[i-1] in self.uniwords:
            unicount = self.uniwords[sentence[i-1]]
        
        bitok = sentence[i-1] + ' ' + sentence[i]
        if bitok in self.biwords:
            bicount = self.biwords[bitok]
        
        score += math.log( (bicount+1) / ( float(unicount) + len(self.uniwords.keys()) ) )
   
    return score

