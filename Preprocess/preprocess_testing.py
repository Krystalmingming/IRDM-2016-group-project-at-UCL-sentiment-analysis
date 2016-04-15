import glob, os
import json
def readTweets():
	path = "/Users/Krystal/Documents/UCL/information retrieval/coursework/group project/Twitter"
	#testFile = open("/Users/Krystal/Desktop/Twitter_test_json.txt", "a")
	for file in os.listdir(path):
		if file.endswith(".txt"):
			dir_entry_path = os.path.join(path, file)
			tweetsFile = open(dir_entry_path,"rU")
			doc = json.load(tweetsFile)
			testFile = open("/Users/Krystal/Desktop/Twitter_test_json_text.txt", "a")
			#testFile_text = open("/Users/Krystal/Desktop/Twitter_test_json_text.txt", "a")
			a = doc['tweets']
			for i in range(0,len(a)):
				b = a[i]
				text = b['text']
				date = b['date']
				print(b['text'])
				print(b['date'])
				#testFile.write('%r\t%r\n' % (date,text.encode('utf-8')))
				#testFile.write('%r\n' % (date))
				testFile.write('%r\n' % (text.encode('utf-8')))
				#testFile_date.write(date)
				#testFile_text.write(text.encode('utf-8'))
				#testFile.write(text.encode('utf-8')+ " ")

if __name__ == '__main__':
	readTweets()
