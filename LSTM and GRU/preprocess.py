import csv

def readUnsupTweets(path):

    tweetsFile=open(path,"r")
    #reader=csv.reader(tweetsFile)
    lines=tweetsFile.readlines()
    
    store_path="./raw_data/train/unsup/"

    counter=0
    for line in lines:
        
        if counter<50000:
            file=open(store_path+"unsup_"+str(counter)+".txt","w")
            file.write(line)
            file.close

        else:    
            break;

        counter+=1    

def readTrainTweets(path):
    # type: (object) -> object
    tweetsFile = open(path, "rb" )
    reader = csv.reader(tweetsFile)

    counter=0
    #pos_counter=0
    #neg_counter=0
    
    for line in reader:

    	if counter>200000:
    		break

        if counter < 100000:
            store_path="./raw_data/train"
        else:
            #counter = 0
            #pos_counter=0
            #neg_counter=0
            store_path="./raw_data/test"

        if line[1] == '1':
            positiveFile = open(store_path+"/pos/pos_"+str(counter)+".txt", "w")
            print(store_path+"/pos/pos_"+str(counter)+".txt")
            positiveFile.write(line[3].lstrip()+'\n')
            positiveFile.close()
            #pos_counter+=1

        else:
            if line[1] == '0':
                negativeFile = open(store_path+"/neg/neg_"+str(counter)+".txt", "w")
                print(store_path+"/neg/neg_"+str(counter)+".txt")
                negativeFile.write(line[3].lstrip()+'\n')
                negativeFile.close()
                #neg_counter+=1

        counter+=1

if __name__ == '__main__':

    file_path="./data/Sentiment_Analysis_Dataset.csv"
    readTrainTweets(file_path)

    unsup_path="./data/Twitter_test.txt"
    readUnsupTweets(unsup_path)


