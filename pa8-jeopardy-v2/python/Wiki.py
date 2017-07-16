# -*- coding: utf-8 -*-

import sys, traceback
import re

class Wiki:
    
    # reads in the list of wives
    def addWives(self, wivesFile):
        try:
            input = open(wivesFile)
            wives = input.readlines()
            input.close()
        except IOError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            sys.exit(1)    
        return wives
    
    def processInfoBox(self,f):
        
        info_on = False
        wife2hubby = {}
        
        for line in f:
                        
            if info_on == False:            
                infobox_matches = re.findall('\{\{Infobox', line)
                if len(infobox_matches) > 0:
                    info_on = True
                    hubby = ""                   
                        
            if info_on:
                infobox_matches = re.findall('^\}\}', line)
                if len(infobox_matches) > 0:
                    #print "!!!!!                OFF", line
                    info_on = False
            
            if info_on:
                           
                matches = re.findall('\|[\s]*[nN]ame[\s]+=[\s]+([A-Za-z-\.\s]+)', line)
                if len(matches) > 0:
                    #print line,matches
                    hubby = matches[0]
                    
                if len(re.findall('[sS]pouse', line)) > 0:    
                    #print line     
                    #matches = re.findall('\[\[([A-Za-z-\s]+)\]\]', line)
                    matches = re.findall('(\w+ \w+)', line)
                    
                    if len(matches) > 0:
                        #print line, "h>",hubby, "w>", matches
                        for m in matches:
                            if m not in wife2hubby:
                                wife2hubby[m] = []
                            
                            wife2hubby[m].append(hubby)
        return wife2hubby
    
    def processTextPattern(self, f):        
        # X is married to Y
        # X was married to Y
        # X married Y
        # X married, xxxx, (\w+ \w+)
        # Y, whom he married
        # [Mm]arriage to (\w+ \w+)
        # Y ]] [.,] [mM]arried
        wife2hubby = {}
         
        person = "[\[\[]*([A-Z][a-z]+\s+[A-Z][a-z]+)[\]\]]*"
        
        #test = "Ingrid Selberg is married to playwright [[Mustapha Matura]]."
        #test = "Ingrid Selberg, who is married to playwright [[Mustapha Matura]]."
         
        #matches = re.findall(person + "[\w\s,]*\s+[wa|i]s\s+married\s+to\s+[\w\s,]*" + person, test)
        #print "test", matches
        
        hubby = ""
        for line in f:
            
            title = re.findall("^'''([\w\s]+)'''", line)
            if len(title) > 0:
                hubby = title[0]
                #print "h=>",hubby
            
            matches = re.findall("[wa|i]s\s+married\s+to[\w\s,]+" + person, line)
            if len(matches) > 0:
                #print "1", matches
                for m in matches:
                    if m not in wife2hubby:
                        wife2hubby[m] = []                        
                    wife2hubby[m].append(hubby)
                
            matches = re.findall("married[\w\s,]+" + person, line)
            if len(matches) > 0:
                #print "2", matches
                for m in matches:
                    if m not in wife2hubby:
                        wife2hubby[m] = []                            
                    wife2hubby[m].append(hubby)
        
        #print wife2hubby
        return wife2hubby
    
    # read through the wikipedia file and attempts to extract the matching husbands. note that you will need to provide
    # two different implementations based upon the useInfoBox flag. 
    def processFile(self, f, wives, useInfoBox):
                        
        husbands = [] 
        
        # TODO:
        # Process the wiki file and fill the husbands Array
        # +1 for correct Answer, 0 for no answer, -1 for wrong answers
        # add 'No Answer' string as the answer when you dont want to answer
        
        if useInfoBox:
            wife2hubby = self.processInfoBox(f)        
        else:
            wife2hubby = self.processTextPattern(f)
                
        
        for w in wives:            
            w = w.replace('\n', '')
            names = w.split(' ')
            full = names[0] + ' ' + names[1]
            if full in wife2hubby:
                h = "Who is " + wife2hubby[full][0].replace('\n','') + "?"                
            else:
                h = 'No Answer'
            
            husbands.append(h)
            print w,"=>" , h
        
        f.close()
        
        return husbands
    
    # scores the results based upon the aforementioned criteria
    def evaluateAnswers(self, useInfoBox, husbandsLines, goldFile):
        correct = 0
        wrong = 0
        noAnswers = 0
        score = 0 
        try:
            goldData = open(goldFile)
            goldLines = goldData.readlines()
            goldData.close()
            
            goldLength = len(goldLines)
            husbandsLength = len(husbandsLines)
            
            if goldLength != husbandsLength:
                print('Number of lines in husbands file should be same as number of wives!')
                sys.exit(1)
            for i in range(goldLength):
                if husbandsLines[i].strip() in set(goldLines[i].strip().split('|')):
                    correct += 1
                    score += 1
                elif husbandsLines[i].strip() == 'No Answer':
                    noAnswers += 1
                else:
                    wrong += 1
                    score -= 1
        except IOError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
        if useInfoBox:
            print('Using Info Box...')
        else:
            print('No Info Box...')
        print('Correct Answers: ' + str(correct))
        print('No Answers: ' + str(noAnswers))
        print('Wrong Answers: ' + str(wrong))
        print('Total Score: ' + str(score)) 

if __name__ == '__main__':
    wikiFile = '../data/small-wiki.xml'
    wivesFile = '../data/wives.txt'
    goldFile = '../data/gold.txt'
    useInfoBox = False;
    wiki = Wiki()
    wives = wiki.addWives(wivesFile)
    husbands = wiki.processFile(open(wikiFile), wives, useInfoBox)
    wiki.evaluateAnswers(useInfoBox, husbands, goldFile)