import glob, os
import collections
import numpy

def readDescriptions():
	pathName = "/Volumes/Kingston/eval/0"
	dict = collections.OrderedDict()
	FullWords = open('words.txt', 'w')
	nextWordIndex = 0
	wordCount = []
	for root, dirs, files in os.walk("/Volumes/Kingston/eval"):
    		for file in files:
    			if file.endswith(".txt"):
    				# print(os.path.join(root, file))
    				txt = open(os.path.join(root, file))
    				lines = txt.readlines()
    				for line in lines:
    					# print line
    					words = line.split(' ')
    					for word in words:
    						cleanedWord = ''.join(list(filter(str.isalnum, word)))
    						cleanedWord = cleanedWord.lower()
    						FullWords.write(cleanedWord + '\n')
    						wordInd = dict.get(cleanedWord, None)
    						if not wordInd:
    							dict[cleanedWord] = 1
    							print(cleanedWord)
    							print(nextWordIndex)
    							wordCount.append(1)
    							nextWordIndex += 1
    						else:
    							wordCount[wordInd] += 1
    							dict[cleanedWord] += 1
	print(dict) 
	out_file = open('dict.txt', 'w')
	for key in dict:
		out_file.write(key + ',' + str(dict[key]) + '\n')
	out_file.close()
	FullWords.close()
	return wordCount
	
	
    						

words = readDescriptions()
print(words)
print(len(words))
