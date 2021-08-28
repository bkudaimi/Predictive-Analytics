#The goal of this assignment is to find out through EDA and regression which days 
#are the best for running a marketing campaign to increase game attendee number.
#We have a dataset of LA Dodgers games with information such as month, day, daily
#temperature, game opponent, weather, number of attendees, and whether items such 
#as caps, shirts, fireworks, and bobbleheads are sold at the game.

#We will run the marketing campaign on days with higher attendee number to increase
#the campaign's audience. To find out which days have the highest attendee number, we will use 
#regression to tell us which features of this dataset contribute the most to attendee number, 
#and from this, we can tell what days (e.g. days in March, Saturdays, etc) would be the best 
#to run our campaign.

#EDA will be conducted to gain insight into the data, then multiple regression will be conducted
#to find out which features of the dataset weigh the most in predicting game attendee number.

#Setting the working directory
setwd("C:/Users/PS3ma/Documents/Bellevue University/DSC 630")

#Importing the necessary libraries
library(Hmisc)
library(ggplot2)
library(QuantPsyc)
library(bootstrap)
#Importing the data
dodgers <- read.csv('dodgers.csv')

#Checking if there is any missing data
is.null(dodgers)


#Viewing the data structure and summary statistics
str(factor(dodgers))
describe(factor(dodgers))

#Generating histograms and bar charts of the key variables.
#In addition to histograms, boxplots of the numerical variables will be generated.
#This will reveal if there is any skewness among the variables.
ggplot(dodgers, aes(day)) + geom_histogram()
ggplot(dodgers, aes(attend)) + geom_histogram()
ggplot(dodgers, aes(temp)) + geom_histogram()
ggplot(dodgers, aes(opponent)) + geom_bar()
ggplot(dodgers, aes(skies)) + geom_bar()
ggplot(dodgers, aes(day_night)) + geom_bar()
ggplot(dodgers, aes(cap)) + geom_bar()
ggplot(dodgers, aes(shirt)) + geom_bar()
ggplot(dodgers, aes(fireworks)) + geom_bar()
ggplot(dodgers, aes(bobblehead)) + geom_bar()
ggplot(dodgers, aes(temp)) + geom_boxplot()
ggplot(dodgers, aes(attend)) + geom_boxplot()
ggplot(dodgers, aes(day)) + geom_boxplot()
#There does not appear to be any skewness, however, most of the games were played on clear nights 
#and souvenirs such as caps, shirts, fireworks, and bobbleheads weren't sold at a majority of games

#Generating a scatter plot of game attendees vs temperature and day of the month to see if 
#temperature or day affects the number of attendees
ggplot(dodgers, aes(temp, attend)) + geom_point()
ggplot(dodgers, aes(day, attend)) + geom_point()

#It doesn't look like temperature or day number is correlated with number of attendees, so the other variables will be 
#plotted against attendee number to see if there are any correlations. Boxplots will be used since the remaining 
#variables to be plotted are categorical in nature.
ggplot(dodgers, aes(skies, attend)) + geom_boxplot()
ggplot(dodgers, aes(opponent, attend)) + geom_boxplot()
ggplot(dodgers, aes(day_night, attend)) + geom_boxplot()
ggplot(dodgers, aes(cap, attend)) + geom_boxplot()
ggplot(dodgers, aes(shirt, attend)) + geom_boxplot()
ggplot(dodgers, aes(fireworks, attend)) + geom_boxplot()
ggplot(dodgers, aes(bobblehead, attend)) + geom_boxplot()
ggplot(dodgers, aes(day_of_week, attend)) + geom_boxplot()


#It appears that Tuesdays have the highest rate of attendance, and that bobbleheads and shirts present 
#at a game are correlated with high attendance. It also appears that games vs the Angels, Mets, and Nationals 
#garner the highest median attendee number, while games vs the Braves and the Pirates garner the lowest median 
#attendee number. Based on the EDA alone, it appears that games vs the Pirates and Braves where shirts and bobbleheads 
#are sold would be a good target for a marketing campaign.

#Multiple linear regression will now be set up to determine which of the factors weigh into game attendee number the most. 
#Any days of the week identified as having a significant weight in attendee number will be reported. 
#First, though, it must be determined that none of the numeric variables are correlated with each other.
#Variables that have correlations with each other, or that have no correlation with the target variable 
#(attendee #number) will be dropped.
cor(dodgers[sapply(dodgers,is.numeric)])

#None of the numeric features appear to be correlated with each other, so we can continue with building the model. 
#Since temperature, day number, and skies did not appear to be correlated with attendee number, these variables 
#will not be used in the regression model.

model <- lm(attend ~ month + day_of_week + opponent + cap + shirt + fireworks + bobblehead, data = dodgers)
summary(model)

#The model coefficients and significance codes will reveal how much weight each variable has in predicting attendee numbers.
#Based on these coefficients in the regression model summary, the days with the highest attendee number are Sundays and Tuesdays.
#In addition, any day when fireworks and bobbleheads are being sold have high attendee numbers. This matches the results of the EDA,
#although opponent did not weigh heavily into the regression model. In addition, the multiple R-squared value was 0.703, meaning that
#this model is a good fit for this data, and the p-value was 3.7e-5, meaning it is highly unlikely this model was generated by chance.
#It is confirmed, then, that the best days to run our marketing campaign to spread it to a wider audience are Sundays and Tuesdays, 
#as well as any day of the week when fireworks and bobbleheads are sold.

