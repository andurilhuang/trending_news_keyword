#!/usr/bin/python3

import datetime
import time
import csv
import pickle
import pandas as pd
import numpy as np
from itertools import islice
from math import sqrt
import math

def zscore(obs, pop):
    '''
    This function calculates z score, which meansures the trend for a keyword, 
    keywords with higher z score are more popular compared to others
    input:  obs is the first item in a list, which should be an interger,
            in this project it represent the the number of articles that mention a keywrod in the most recent interval
            within the current date. 
            pop is a list, which contains integers represents the number of articles that mention
            the keyword in the consecutive intervals
    output: The function returns a float. 
    '''
    # Size of population.
    number = float(len(pop))
    # Average population value.
    avg = sum(pop) / number
    # Standard deviation of population.
    std = sqrt(sum(((c - avg) ** 2) for c in pop) / number)
    # Zscore Calculation.
    if std == 0:
        return obs
    else:
        return (obs - avg) / std

def cleanData(df):
    '''
    This function reads in a csv file and convert it into a list, and delete the title row and rows with 
    no date information.
    input: a csv file, the 4th column is the date information and the 12th column is the keyword data.
    output: a list object
    '''     
    #csv_reader = csv.reader(open(csvFile, encoding='utf8'))
    data = pd.read_pickle(df)
    #convert to list, each row in the csv is an item.
    #rows = list(csv_reader)
    rows = data.values.tolist()
    #delete the title row
    #del rows[0]
    #Delete rows with no date information
    for i in range(len(rows)-1,-1,-1):
        date = rows[i][3]
        if date == '':
            del rows[i]
    return rows

def convertDateFormat(rows):    
    '''
    Convert the date in a list of data into yyyy/mm/dd format and sort data in date reverse order,
    and also delete text column data.
    input: a list of which each item is also a list that represents one row of data in the csv
    output: a list object.
    '''        
    #Convert dates to format yyyy/mm/dd and delete text column.  
    for i in range (len(rows)):
        timeString = rows[i][3]
        date = timeString.split(' ')[0]
        # convert all dates to yyyy/mm/dd format
        #find:return -1 if '-' not found in the string, which is yyyy/mm/dd format therefore 
        # doesn't need format conversion.
        if date.find('-') != -1:
            dateList = date.split('-')
            convertedDateFortmat = '/'.join(dateList)#yyyy/mm/dd
            rows[i][3] = convertedDateFortmat
        else:
            rows[i][3] = date
        dateSegments = rows[i][3].split('/')#['2016','9','30']
        month = dateSegments[1]
        date = dateSegments[2]
        if int(month)<10 and len(month)<2:
            newMonth = '0'+ month
            dateSegments[1] = newMonth
        if int(date)<10 and len(date)<2:
            newDate = '0'+ date
            dateSegments[2] = newDate
        newDateWithZero = '/'.join(dateSegments)#2016/09/30
        dateTimeFormat = datetime.datetime.strptime(newDateWithZero,'%Y/%m/%d')#2016-09-03 00:00:00
        rows[i][3] = dateTimeFormat
        #delete the text column to make the output more clear.
        #del rows[i][2]
    #sort all info in csv in time reverse order
    rows.sort(key=lambda item: (item[3]), reverse=True)
    return rows

def getEachKwHistoricalCountsDaily(rows, duration):
    '''
    This function calculate the number of articles that mention each keyword on each day 
    within a given period.
    input:  rows: a list of which each item is a also a list contains keyword information , the 3rd column should 
            be keyword's date and the 12th column should be the keyword or a list of keywords.
            duration should be an integer in range 1 to 13.
    output: a dictionary with keys represent keyword, values are lists represent number of articles that 
            mention the keyword on each day within the given period.
    '''
    durationInFormat = datetime.timedelta(days = duration)
    for row in rows:# row1
        keywords = []
        for content in row[11].strip("[]'").split("', '"):
            keywords.append(content)
        for keyword in keywords:
            keywordDate = row[3]
            # If this keyword is published in article within the periods we want to see
            if (keywordDate >= currentDate - durationInFormat) and (keywordDate <= currentDate -datetime.timedelta(days = 1)):
                countLocationInList = (currentDate - keywordDate - datetime.timedelta(days = 1)).days
                if keyword in dict:
                    allInfo = dict[keyword]
                    allInfo['articleCoverage'][countLocationInList] += 1
                    helper(allInfo,row)
                else:
                    # initialize a list with each item equals to 0.
                    list = []
                    for i in range(duration):
                        list.append(0)
                    list[countLocationInList] += 1
                    allInfo = {'articleCoverage':list,'text_clean': [],'text_raw':[],'timestamp':[],'title':[],'url':[],'entity': [],'entityID':[]}
                    helper(allInfo,row)
                    dict[keyword] = allInfo
    return dict

def getEachKwHistoricalCountsWeekly(rows, duration):
    '''
    This function calculate the number of articles that mention each keyword in each week 
    within a given period, if whole week can not be obtained within the duration given (i.e.22 days) the
    remaining days data will not be considered (i.e. only consider 21 days if duration = 22 )
    input:  rows: a list of which each item is a also a list contains keyword information , the 3rd column should 
            be keyword's date and the 12th column should be the keyword or a list of keywords.
            duration should be an integer > 13.
    output: a dictionary with keys represent keyword, values are lists represent number of articles that 
            mention the keyword in each week within the given period.
            For example if currentDate is set as Sep 1st, 2016 and duration is set as 22 days, than an entry for keyword
            "technology" in the dictionary should be "technology":[3,2,1], 3 represents article coverage in 8.24~8.30, 
            2 represents article coverage in 8.17~8.23, and 3 represents article coverage in 8.10~8.16
    '''
    durationInFormat = datetime.timedelta(days = duration)
    numberOfWeeks = int(duration / 7) #4
    for row in rows:
        keywords = []
        for content in row[11].strip("[]'").split("', '"):
            keywords.append(content)
        for keyword in keywords:
            keywordDate = row[3]
            startDate = currentDate - datetime.timedelta(days = 7*numberOfWeeks)
            endDate = currentDate - datetime.timedelta(days = 1)
            if (keywordDate >= startDate) and (keywordDate <= endDate):
                diff = (currentDate - keywordDate).days #14
                locationInList = math.ceil(diff/7) - 1 #1
                if keyword in dict:
                    allInfo = dict[keyword]
                    allInfo['articleCoverage'][locationInList] += 1
                    helper(allInfo,row)
                else:
                    list = []
                    for i in range(numberOfWeeks):
                        list.append(0)
                    list[locationInList] += 1
                    #create a map to store all information for a trending keyword
                    allInfo = {'articleCoverage':list,'text_clean': [],'text_raw':[],'timestamp':[],'title':[],'url':[],'entity': [],'entityID':[]}
                    helper(allInfo,row)
                    dict[keyword] = allInfo
    return dict

def helper(dict,row):
    dict['text_clean'].append(row[1])
    dict['text_raw'].append(row[2])
    dict['timestamp'].append(row[3])
    dict['title'].append(row[4])
    dict['url'].append(row[5])
    dict['entity'].append(row[6])
    dict['entityID'].append(row[7])
    
def saveKeywordInfoIntoDict(rows):
    '''calculate article coverage on weekly time interval if the duration >= 14 days
        otherwise calculate article coverage on daily time interval
    input: takes in a list contains all info in article database and returns a dictionary
            with keys represent keyword, values are lists represent number of articles that 
            mention the keyword in each time interval within the given period.
    '''
    if (duration < 14):
        dict = getEachKwHistoricalCountsDaily(rows, duration)
    else:
        dict = getEachKwHistoricalCountsWeekly(rows, duration)
    return dict


def writeOutputCsvTitle(dict):
    '''
    Write results to the output file
    input: fileName is the output csv file name, should be a string, end up in .csv
            interval should be 'day' or 'week'.
            data: a dictionary, keywords are contained as keys and list as values, the list contains
            the number of articles which mention this keyword in date descending order.
    '''
    # csvOutputFile = open(fileName, 'w', encoding='utf8',newline='')
    # writer = csv.writer(csvOutputFile)
    # write title row into csv file
    titleList = ['Keyword']
    allInfoPrintToCsv = []
    # if calculate counts in daily frequency 
    if duration < 14:
        for i in range(1, duration + 1):
            if i == 1:
                title = str(i) + ' day ago'
            else: 
                title = str(i) + ' days ago'
            titleList.append(title)
        titleList.append('Z-score')   
        # writer.writerow(titleList)
        # write content to csv file
        for keyword in dict:
            oneLinePrintToCsv = []
            oneLinePrintToCsv.append(keyword)
            for i in range(0,duration):
                oneLinePrintToCsv.append(dict[keyword]['articleCoverage'][i])
            z = zscore(dict[keyword]['articleCoverage'][0], dict[keyword]['articleCoverage'][1:duration])
            oneLinePrintToCsv.append(z)
            dict[keyword]['z-score'] = z
            allInfoPrintToCsv.append(oneLinePrintToCsv)    
        allInfoPrintToCsv.sort(key=lambda item: item[duration + 1], reverse=True) 
        for i in range(0, min(100, len(allInfoPrintToCsv))):
            addInfoIntoResultHelper(result,dict,allInfoPrintToCsv[i][0])
    #        writer.writerow(allInfoPrintToCsv[i])
    # if calculate counts in weekly frequency 
    else:
        numberOfWeeks = int(duration / 7)
        for i in range(1, numberOfWeeks + 1):
            if i == 1:
                title = str(i) + ' week ago'
            else: 
                title = str(i) + ' weeks ago'
            titleList.append(title)
        for item in ['Z-score','text_clean','text_raw','timestamp','title','url','entity','entityID']:
            titleList.append(item)   
        #writer.writerow(titleList)
        for keyword in dict:
            oneLinePrintToCsv = []
            oneLinePrintToCsv.append(keyword)
            for i in range(0,numberOfWeeks):
                oneLinePrintToCsv.append(dict[keyword]['articleCoverage'][i])
            z = zscore(dict[keyword]['articleCoverage'][0], dict[keyword]['articleCoverage'][1:numberOfWeeks])
            oneLinePrintToCsv.append(z)
            dict[keyword]['z-score'] = z
            allInfoPrintToCsv.append(oneLinePrintToCsv)  
        allInfoPrintToCsv.sort(key=lambda item: item[numberOfWeeks + 1], reverse=True)
        for i in range(0, min(100, len(allInfoPrintToCsv))):
            #writer.writerow(allInfoPrintToCsv[i])
            addInfoIntoResultHelper(result,dict,allInfoPrintToCsv[i][0])
    return result
    
def addInfoIntoResultHelper(result,dict,keyword):
    result['Trending_keyword'].append(keyword)
    result['articleCoverage'].append(dict[keyword]['articleCoverage'])
    result['z-score'].append(dict[keyword]['z-score'])
    result['text_clean'].append(dict[keyword]['text_clean'])
    result['text_raw'].append(dict[keyword]['text_raw'])
    result['timestamp'].append(dict[keyword]['timestamp'])
    result['title'].append(dict[keyword]['title'])
    result['url'].append(dict[keyword]['url'])
    result['entity'].append(dict[keyword]['entity'])
    result['entityID'].append(dict[keyword]['entityID'])

def getArticleCoverageForKeyword(keyword, days, currentDate):#2018/04/23
    '''
    Get each day's number of articles for a given keyword within a given period of time
    For example an input of ('car',5, 20180423) will show the number of articles that mention keyword 'car'
    in the most recent 5 days for date 2018/04/23
    input: keyword should be a string, days should be an integer, currentDate should be in the format of 
            YYYYMMDD.
    output: the function will return a list with two items, of which the first item is a list contains
            all dates, and the second is a list containing number of appearance in articles published in the respective date.
    '''
    currentDateInDateFormat = datetime.datetime.strptime(str(currentDate),'%Y%m%d')
    i = 0
    dateList = []
    countList = []
    while(i < days):
        date = currentDateInDateFormat - datetime.timedelta(days = i+1)#2018/04/23 datetime format
        if date in dict:
            KeywordListsAtThatDate = dict[date]
            wordCount = 0
            for keywordList in KeywordListsAtThatDate:
                keywordListForOneArticle = keywordList.strip("[]'").split("', '")
                for word in keywordListForOneArticle:
                    if keyword == word:
                        wordCount += 1
            dateList.append(date)
            countList.append(wordCount)
        i += 1
    result = [dateList,countList]
    return result

def plotDateAndArticleCoverage(keyword, period, currentDate):
    '''
    This function plot a line chart for a given keyword's daily article coverage within a given
    time period
    Input: keyword: a string
            period: an integer
            currentDate: yymmdd format
    '''
    data = getArticleCoverageForKeyword(keyword, period, str(currentDate))
    date = data[0]
    coverage = data[1] 
    plt.plot(date,coverage) 
    #set lables  
    plt.xlabel("Dates")  
    plt.ylabel("Number of articles")  
    plt.title("Recent "+ str(period) +" days trend for " + '"'+keyword+'"')
    #set x label ticks rotation
    plt.xticks(rotation=40)
    firstDate = date[len(date)-1]
    lastDate = date[0]
    plt.xlim(firstDate, lastDate)
    #set y label ticks number of interval
    y_ticks = np.linspace(0, max(coverage), max(coverage)+1)
    plt.yticks(y_ticks) 
    plt.show()


dict = {}
result = {'Trending_keyword': [], 'articleCoverage':[],'z-score':[],
          'text_clean': [],'text_raw':[],'timestamp':[],'title':[],
          'url':[],'entity': [],'entityID':[]}
currentDate = datetime.datetime(2017, 1, 6, 0, 0, 0)
duration = 28
data = cleanData('keyword_data_final2.pkl')#list
sortedData = convertDateFormat(data)#list
resultDict = saveKeywordInfoIntoDict(sortedData) #get dict
writeOutputCsvTitle(resultDict)
result_df = pd.DataFrame(result)
result_df.to_pickle('finalResult.pkl') 





















