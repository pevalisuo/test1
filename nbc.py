#!/usr/local/python
#
# This program learns statistical dependecies between
# variables (columns) in a CSV file and can after that
# find out most probable new records. Naiive Bayes
# classifier is used for this purpose.
# 
# $Id: nbc.py,v 1.5 2002/04/06 13:02:14 petri Exp petri $
# $Log: nbc.py,v $
# Revision 1.5  2002/04/06 13:02:14  petri
# Continuing towards multiple free states. Does not work
# extremely well yet
#
#
import re
import xreadlines
#import fileinput

#============================================================
# Naive bayes classifier class
#============================================================
class naiveBayes:
    """
    Make naive bayes classification for CSV data read from a file
    """

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def __init__(self):
        """ Initialize the variables """
        self.filename=""
        self.labels={}
        self.features=[]
        self.freq=[]
        self.prior={'M': 0, 'p': 0}
        self.filter=None

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def reset(self):
        """ Reset variables """
        self.filename=""
        self.labels={}
        self.features=[]
        self.freq=[]
        self.prior={'M': 0, 'p': 0}
        self.filter=None

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def read(self, filename, delim=",", filter=None):
        """ Read the labels and features from CSV file """

        # Define the regular expression to delimit words
        self.delim=re.compile("\s*%s\s*" % (delim))

        # Open the file
        self.filename=filename
        fid=open(filename,"r")

        # Read labels, and number them. Initialize also a data
        # structure for frequencies
        labels=self.delim.split(fid.readline().strip())
        i=0
        for fn in labels:
            self.labels[fn]=i
            self.freq.append({})
            i+=1
        n=len(self.labels)

        # Read feature vectors from the file
        while 1:
            line=fid.readline()
            if not line: break
            line=line.strip()

            # Skip empty lines
            if(len(line)==0):
                continue
            
            features=self.delim.split(line)
            filterOut=0
            if filter:
                for k in filter.keys():
                    if features[self.labels[k]]!=filter[k]:
                        filterOut=1
            if(n==len(features) and not filterOut):
                self.features.append(features)
            else:
                print "Illegal line or filtered away: ", line


    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def getFields(self):
        """ Return the labels of the columns as a vector"""
        return self.labels
    
    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def getStates(self, fieldname):
        """ Return the names fo the states of the given column (feature)"""
        fieldno=self.labels[fieldname]
        return self.freq[fieldno].keys()
    
    
    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def setUniformPrior(self, eqs_size, propability):
        self.prior['M']=eqs_size
        self.prior['p']=propability

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def setFilter(self, filter):
        """ Sets filter for read features. The filter consists of
        n tuples. Each tuple contains three fields 1: the field
        name, 2: the field value 3: 1 or 0, 0=exclude and 1= include.

        Inclusive filter leaves only those records whose named field
        matches the value.

        Exclusive filter leaves only those records whose named field
        does not match the value."""
        self.filter=filter

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def getConditional(self, A, B):
        """Calculate conditional probability
           P(A=a|B=b) = F(A=a and B=b)/F(B=b)

           Add also prior estimate
           P(A=a|B=b) = (F(A=a and B=b) + M*p) / ( F(B=b) +M )

           Where M is equivalent sample size of prior estimate
           and p is the prior probability
           """

        ia=self.labels[A[0]]
        ib=self.labels[B[0]]

        # Calculate frequency F(A=a and B=b)
        freqAB=0.0
        for record in self.features:

            # Filter out unmatching records if filter is defined
            filterOut=0
            if self.filter:
                for k in self.filter:
                    label=k[0]
                    if record[self.labels[label]]==k[1]:
                        filterOut=not k[2]
                    else:
                        filterOut=k[2]
                    if(filterOut):
                        break

            # Increment conditional frequency counter if both
            # record matches both A and B
            if not filterOut:
                if(record[ia]==A[1]
                   and record[ib]==B[1]):
                    freqAB+=1

        # Get frequency F(B=b)
        freqB=float(self.freq[ib][B[1]])

        # Calculate the conditional propability
        P=(freqAB+self.prior['M']*self.prior['p'])/(freqB + self.prior['M'])
        
        return P

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def frequencies(self):
        """ Calculate the frequencies of the states of all features
        A example of the frequency table is following:

        Column
        1     { eka:4,  toka:26,  kolmas:4 }
        2     { jaa:5, ei:150, tyhjä:10, poissa:35 }
        3     { red:3, green:2, blue:0 }
        4

        """

        n=0
        for record in self.features:

            # Filter out unmatching records if filter is defined
            filterOut=0
            if self.filter:
                for k in self.filter:
                    label=k[0]
                    if record[self.labels[label]]==k[1]:
                        filterOut=not k[2]
                    else:
                        filterOut=k[2]
                    if(filterOut):
                        break

            # Increment the appropriate frequency fields
            if not filterOut:
                n+=1
                for i in range(len(record)):
                    if(self.freq[i].has_key(record[i])):
                        self.freq[i][record[i]]+=1
                    else:
                        self.freq[i][record[i]]=1
                        
        return n

    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def probOrder(self,x,y):
        return cmp(y[1], x[1])

    # ------------------------------------------------------------
    # Find maximum probability combination of free parameters
    # ------------------------------------------------------------
    def maxProb(self, state, nresults=1):
        """Find a values for free parameters which forms a
        combination whose probability is largest, using well
        known naive Bayes formula:

           P(f|x1,x2,...,xn) = P(x1, x2,...,xn|f) * P(f) / P(x1,x2,...,xn)

        Using independence assumption of naive Bayes classifier:

           P(f|x1,x2,...,xn) = prod{ P(xi|f) } * P(f) / prod P(xi)

        The value f having maximum probability is therefore achieved
        by maximizing the value:
        
           prod { P(xi|f) } P(f)
        """

        # Free variables are indicated as None values, Replace Nones
        # with all possible states
        freestates=[]
        fixedstates=[]
        for var in state.keys():
            if(state[var]==None):
                state[var]=self.getStates(var)
                freestates.append(var)
            else:
                fixedstates.append(var)

        
        #print "DEBUG>> States:", state
        #print "DEBUG>> Free:", freestates
        #print "DEBUG>> Fixed:", fixedstates

        # Find out number or different combinations
        N=1
        for var in state.keys():
            N*=len(state[var])
            

        # Initilaize propabilties
        probs={}
        probsum=0.0
        combs=[]

        # Go through all possible combinations
        for i in range(N):
            div=1
            comb={}
        
            # Choose a combination number i, by selecting a value
            # for each variable

            index=i
            for var in state.keys():
                mod=len(state[var])
                j=i/div%mod
                div*=mod
                comb[var]=state[var][j]

            combs.append(comb.values())
                

            # Go through each free state
            for freeS in freestates:
                
                # Calculate P(freeS), ie A Priori propability
                fstateno=self.labels[freeS]
                n=float(self.freq[fstateno][comb[freeS]])
                prob=n/float(len(self.features))

                # Go through each fixed state to calculate conditionals
                for fixedS in comb.keys():

                    # Exclude the free state
                    if(fixedS==freeS):
                        continue

                    # Calculate P(xi|freeS)
                    cond=self.getConditional((fixedS, comb[fixedS]),
                                             (freeS ,comb[freeS]))

                    #print ">>> Cond = ", cond
                    if(probs.has_key(index)):
                        probs[index]*=prob*cond
                    else:
                        probs[index]=prob*cond

	    if(probs.has_key(index)):
               probsum+=probs[index]
            else:
               print "Don't have key", index

        # Normalize probabilities
        sortedProbs=[]
        for k in probs.keys():
            probs[k]/=probsum
            sortedProbs.append((combs[k], probs[k]))

        # Sort by ascending propability
        #sortedProbs=probs.items()
        sortedProbs.sort(self.probOrder)
        
        return sortedProbs[0:nresults]


    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------
    def debug(self):
        """Print some information of the internal state of the object
        for debugging purposes"""
        print self.filename
        print self.labels
        print self.features

    def test(self):
        self.read("f1.txt", ",")
        self.setFilter((("Driver",'MH', 0)
                        , ("Year",'1999', 0)
                        , ("Year",'2001', 0)
                        , ("Year",'2000', 0)))
        print self.frequencies()
        self.setUniformPrior(1,0.5)
        print self.getStates("Year")
        print self.getStates("Driver")
        print self.maxProb({'Driver':['MS'], 'Talli':None},3)

#============================================================
# Main program for testing
#============================================================
if __name__=='__main__':

    print "Start of program"
    b=naiveBayes()
    #b.read("data.txt")
    b.read("f1data.txt", "\t", {"Year": "2001"})
    b.debug()
    print "==================== Fields ======================"
    fn=b.getFields()
    print fn
    print "==================== Frequencies ======================"
    b.frequencies()
    print b.freq
    print "==================== Some states ======================"
    print b.getStates('Year')
    print b.getStates('Position')
    b.setUniformPrior(1,0.5)
    print "================ Some conditional probs  ==================="
    #print b.getConditional(("Position", "1"),("Driver", "RS"))
    print "==================== Max prob  ======================"
    maxprobs=b.maxProb({
        'Driver': ["MS"],
        'Country': ['Malesia'],
        'Position' : None
        }, 3)
    for value in maxprobs:
        print "%20s %20s" % value
    print "End of program"
