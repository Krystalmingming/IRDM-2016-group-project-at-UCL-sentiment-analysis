import datetime
import csv

def readFile(filePath):

    dateSenti = dict()

    file=open(filePath,'r')

    for line in file:

        posArr=[]

        token=line.split("\t")

        timestamp=token[0][1:(len(token[0])-4)]
        #print timestamp

        readableDate=datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')

        sentiment=token[1][1:(len(token[1])-2)]
        #print sentiment

        if(dateSenti.has_key(readableDate)):

            posArr=dateSenti.get(readableDate)
            posArr.append(sentiment)
            dateSenti[readableDate]=posArr

        else:
            posArr.append(sentiment)
            dateSenti[readableDate]=posArr

    return dateSenti


# print(
#     datetime.datetime.fromtimestamp(
#         int("1446333432")
#     ).strftime('%Y-%m-%d %H:%M:%S')
# )

if __name__ == '__main__':
    filePath='./data/ME_output_new.txt'
    storePath='date_sentiment.csv'

    file=open(storePath,'w')
    dateSenti =readFile(filePath)

    for date in sorted(dateSenti.keys()):

        #print date+','.join(dateSenti.get(date))
        sentimentArr=dateSenti.get(date)

        posNum=0
        negNum=0

        for sentiment in sentimentArr:
            if sentiment=='pos':
                posNum+=1
            else:
                if sentiment=='neg':
                    negNum+=1

        p = float(posNum)/negNum
        file.write(date+','+str(p)+'\n')




