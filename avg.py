from csv import reader
import csv


def get_args_parser():
    parser = argparse.ArgumentParser('csv_avg ', add_help=False)
    parser.add_argument('--input', default=" ", type=str)
    parser.add_argument('--output', default=" ", type=str)
    parser.add_argument('--avg', default=1, type=int)
    return parser

parser = argparse.ArgumentParser('option\'s', parents=[get_args_parser()])
args = parser.parse_args()
inputFile=args.input
outputFile=args.output
limit=args.avg

fileName=inputFile
newfileName=outputFile
count =0
avglist=[]
with open(fileName, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    temp=0
    var=0
    count=0
    
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        #print(float(row[0]))
        if count == (limit-1) :
            var=float(row[0])
            temp= var+ temp
            count=0
            avglist.append((temp/limit))
            temp=0
        else:
            var=float(row[0])
            count = count+1
            temp= var+ temp

print(len(avglist))


with open(newfileName, mode='w') as avglistofangle:
    avgList_writer = csv.writer(avglistofangle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(avglist)):
        avgList_writer.writerow([avglist[i]])







