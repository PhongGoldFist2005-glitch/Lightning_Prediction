from datetime import datetime
import re

class classification():
    def __init__(self):
        pass
    def makeTimeOrder(self,input_file,output_file):
        dateList = {}
        with open(input_file,"r") as f:
            for item in f:
                data = re.split("[/:h;]",item.strip())
                year = int(data[0])
                month = int(data[1])
                day = int(data[2])
                hour = int(data[3])
                minute = int(data[4])
                dateObject = datetime(year=year,month=month,day=day,hour=hour,minute=minute)
                band = str(data[6]).strip()
                if band not in dateList:
                    dateList[band] = []
                dateList[band].append(dateObject)
            # print(dateList)
        # sort
        for key in dateList:
            dateList[key].sort()
        with open(output_file,"w") as f:
            for key in sorted(dateList):
                for item in dateList[key]:
                    f.write(f"{item.year}/{item.month}/{item.day}:{item.hour}h{item.minute}; Band Missing: {key}\n")

        
a = classification()
a.makeTimeOrder("/sdd/Dubaoset/src/Phong/Log_file_error/final_2023.txt","/sdd/Dubaoset/src/Phong/Log_file_error/order__2023.txt")
