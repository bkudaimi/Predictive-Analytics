library(rjson)
library(blsAPI)
library(dplyr)
library(Hmisc)
library(ggplot2)
library(gmodels)

#Part 1: Requesting the BLS data

#Requesting the unemployment rate in California
CA_Unemployed <- blsAPI('LASST060000000000003', return_data_frame = TRUE)

#Requesting the employment-population ratio in California
CA_EPRatio <- blsAPI('LASST060000000000007', return_data_frame = TRUE)

#Requesting the export price index of organic chemicals
CA_EPI_org <- blsAPI('EIUID29', return_data_frame = TRUE)

#Merging these three together
CA1 <- merge(CA_Unemployed, CA_EPRatio, by=c("year","periodName"))
CA2 <- merge(CA1, CA_EPI_org, by = c("year", "periodName"))

#Renaming some of the columns
names(CA2)[names(CA2) == "value.x"] <- "CA_Unemployment_Rate"
names(CA2)[names(CA2) == "value.y"] <- "CA_EP_Ratio"
names(CA2)[names(CA2) == "value"] <- "CA_Employment_Price_Index_Chemicals"

#Convert string representations to numbers
CA2$CA_Unemployment_Rate <- as.numeric(CA2$CA_Unemployment_Rate)
CA2$CA_EP_Ratio <- as.numeric(CA2$CA_EP_Ratio)
CA2$CA_Employment_Price_Index_Chemicals <- as.numeric(CA2$CA_Employment_Price_Index_Chemicals)

#Part 1: Summary statistics for 2 variables, histograms, boxplots, density plots, and saving the data as a CSV file

#Summary statistics of 2 variables using the Hmisc library
summary1 <- describe(CA2$CA_Unemployment_Rate)
summary2 <- describe(CA2$CA_EP_Ratio)
summary1
summary2

#Histograms of the 2 variables

ggplot(CA2, aes(x = CA_Unemployment_Rate)) + geom_histogram()
ggplot(CA2, aes(x = CA_EP_Ratio)) + geom_histogram()

#Boxplots of the 2 variables

ggplot(CA2, aes(x = CA_Unemployment_Rate)) + geom_boxplot()
ggplot(CA2, aes(x = CA_EP_Ratio)) + geom_boxplot()

#Density plots of the 2 variables

ggplot(CA2, aes(x = CA_Unemployment_Rate)) + geom_density()
ggplot(CA2, aes(x = CA_EP_Ratio)) + geom_density()

#Saving the data frame locally as a CSV file
write.csv(CA2, 'CA2.csv')

#Part 2: Generating bivariate analyses from the same dataset but with different variables and finding the correlation

#Bivariate plots
ggplot(CA2, aes(x = CA_Unemployment_Rate, y = CA_Employment_Price_Index_Chemicals)) + geom_point()
ggplot(CA2, aes(x = CA_Unemployment_Rate, y = CA_EP_Ratio)) + geom_point()
ggplot(CA2, aes(x = CA_EP_Ratio, y = CA_Employment_Price_Index_Chemicals)) + geom_point()

#Only the plot of EPI vs unemployment rate showed a linear trend, so a Pearson correlation will be used for that relationship only.
#The other variable relationships will have their correlations calculated using the spearman method.
cor(CA2$CA_Unemployment_Rate, CA2$CA_Employment_Price_Index_Chemicals, method = "spearman")
cor(CA2$CA_Unemployment_Rate, CA2$CA_EP_Ratio, method = "pearson")
cor(CA2$CA_EP_Ratio, CA2$CA_Employment_Price_Index_Chemicals, method = "spearman")

#Cross tables
CrossTable(CA2$CA_Unemployment_Rate, CA2$CA_Employment_Price_Index_Chemicals, digits = 2)
CrossTable(CA2$CA_Unemployment_Rate, CA2$CA_EP_Ratio, digits = 2)
CrossTable(CA2$CA_EP_Ratio, CA2$CA_Employment_Price_Index_Chemicals, digits = 2)

#Part 3: Summary report

summary(CA2)
str(CA2)

#See R Markdown report for a discussion of four results of the data