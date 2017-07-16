import math

class CustomLanguageModel:

  # Kneser-Neg
    
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.uniwords = {}
    self.biwords = {} 
    self.startwith = {}
    self.endwith = {}
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    unkct = 0
    
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
                    self.biwords[bitok] += 1
                else:
                    self.biwords[bitok] = 1
                    
                if unitok in self.startwith:
                    self.startwith[unitok] += 1
                else:
                    self.startwith[unitok] = 1
                    
                if sentence.data[i+1].word in self.endwith:
                    self.endwith[sentence.data[i+1].word] += 1
                else:
                    self.endwith[sentence.data[i+1].word] = 1

    
    for w in self.uniwords:
        if self.uniwords[w] == 1:
            self.uniwords[w] = 0
            unkct += 1
    
    self.uniwords["UNK"] = unkct
    #print "finish training"
                

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    
    for i in range(1, len(sentence)):
        unicount = 0.0
        bicount = 0.0
        wf = 0
        wp = 0
        d = 1
        
        if sentence[i-1] in self.uniwords and self.uniwords[sentence[i-1]] > 0:
            unicount = self.uniwords[sentence[i-1]]
        else:
            unicount = self.uniwords["UNK"]
            
        bitok = sentence[i-1] + ' ' + sentence[i]
        if bitok in self.biwords:
            bicount = self.biwords[bitok]            
        
        #print bitok
        
        #The number of word types that can follow wi-1
        if sentence[i-1] in self.startwith:
            wf = self.startwith[sentence[i-1]]

        #The number of word types seen to precede w
        if sentence[i] in self.endwith:
            wp = self.endwith[sentence[i]]
        
        #for w in self.biwords:
        #    if w.startswith(sentence[i-1]):
        #        wf += 1
        #    if w.endswith(sentence[i]):
        #        wp += 1
        
        lamda = d / unicount * wf
        pcont = float(wp) / len(self.biwords.keys())
        
        p = lamda * pcont + max(bicount - d, 0) / float(unicount)
        
        
        if p == 0:      
            #print "aiya",lamda, pcont, p
            p = 10**-20
            
        score += math.log(p)

    #print "returning", score
    return score
