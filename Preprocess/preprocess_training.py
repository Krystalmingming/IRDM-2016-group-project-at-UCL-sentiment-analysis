import csv
def readTweets():
	tweetsFile = file( "/Users/zxy/Desktop/Sentiment_Analysis_Dataset.csv", "rb" )
	reader = csv.reader(tweetsFile)
	positiveFile = open("/Users/zxy/Desktop/positive.txt", "w")
	negativeFile = open("/Users/zxy/Desktop/negative.txt", "w")
	for line in reader:
		if line[1] == '1':
			positiveFile.write(line[3].lstrip()+'\n')

		else:
			if line[1] == '0':
				negativeFile.write(line[3].lstrip()+'\n')
	positiveFile.close()
	negativeFile.close()

if __name__ == '__main__':
	readTweets()


