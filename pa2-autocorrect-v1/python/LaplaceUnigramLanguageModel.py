import math
class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = {}
    self.total = 0
    
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
      for datum in sentence.data:
        self.total = self.total +1
        token = datum.word
        if token in self.words:
            self.words[token] = self.words[token]+1
        else:
            self.words[token] = 1
    self.addOneTotal = float(len(self.words.keys())+self.total)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    for tok in sentence:
      if tok in self.words:
        count = self.words[tok]
      else:
        count = 0 
      score += math.log( (count+1) / self.addOneTotal)

    return score

