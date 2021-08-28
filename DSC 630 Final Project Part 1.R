#DSC 630 Final Project Part 1: Data preparation

#Importing the necessary libraries
library(Hmisc)
library(tidyr)
library(stringr)
library(dplyr)
library(ggplot2)
library(data.table)

#Importing the dataset
df_movies <- read.csv('C:/Users/PS3ma/Documents/Bellevue University/DSC 630/movie_metadata.csv')

#Viewing the data structure
str(df_movies)

#Viewing the number of missing data from each column
print(colSums(sapply(df_movies, is.na)))

#Getting rid of incomplete rows and viewing the new size of the data frame
df_movies <- df_movies[complete.cases(df_movies), ]
print('New size of the data frame:')
print(dim(df_movies))

#Each movie title in the title column has a special character at the end of the title, so it will be removed
df_movies$movie_title <- gsub("Â", "", as.character(factor(df_movies$movie_title)))

#Trimming any whitespaces from each character column
df_movies$movie_title <- trimws(str_trim(df_movies$movie_title, side = "both"))
df_movies$language <- trimws(str_trim(df_movies$language, side = "both"))
df_movies$country <- trimws(str_trim(df_movies$country, side = "both"))
df_movies$content_rating <- trimws(str_trim(df_movies$content_rating, side = "both"))
df_movies$actor_1_name <- trimws(str_trim(df_movies$actor_1_name, side = "both"))
df_movies$actor_2_name <- trimws(str_trim(df_movies$actor_2_name, side = "both"))
df_movies$actor_3_name <- trimws(str_trim(df_movies$actor_3_name, side = "both"))

#Viewing the structure again
str(df_movies)

#Determining if there are any linear correlations
correlations <- cor(df_movies[sapply(df_movies, is.numeric)])
correlations
#There are several highly correlated variables, so they will be removed during the feature selection step later

#One-hot encoding movie genres in a new data frame to be merged with the old data frame
df_genres <- as.data.frame(df_movies[,c("genres", "imdb_score")])

df_genres$Action <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Action") 1 else 0)
df_genres$Adventure <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Adventure") 1 else 0)
df_genres$Animation <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Animation") 1 else 0)
df_genres$Biography <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Biography") 1 else 0)
df_genres$Comedy <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Comedy") 1 else 0)
df_genres$Crime <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Crime") 1 else 0)
df_genres$Documentary <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Documentary") 1 else 0)
df_genres$Drama <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Drama") 1 else 0)
df_genres$Family <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Family") 1 else 0)
df_genres$Fantasy <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Fantasy") 1 else 0)
df_genres$`Film-Noir` <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Film-Noir") 1 else 0)
df_genres$History <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "History") 1 else 0)
df_genres$Horror <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Horror") 1 else 0)
df_genres$Musical <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Musical") 1 else 0)
df_genres$Mystery <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Mystery") 1 else 0)
df_genres$News <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "News") 1 else 0)
df_genres$Romance <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Romance") 1 else 0)
df_genres$`Sci-Fi` <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Sci-Fi") 1 else 0)
df_genres$Short <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Short") 1 else 0)
df_genres$Sport <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Sport") 1 else 0)
df_genres$Thriller <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Thriller") 1 else 0)
df_genres$War <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "War") 1 else 0)
df_genres$Western <- sapply(1:length(df_genres$genres), function(x) if (df_genres[x,1] %like% "Western") 1 else 0)

#Merging the new data frame with the old data frame. The variable imdb_score.x will be used as it is the original
#scores, while imdb_scores.y will be deleted. 
df_movies <- merge(df_movies, df_genres, by = 'genres')

#Dropping duplicated rows
df_movies <- df_movies[!duplicated(df_movies$movie_title), ]

#Defining a new column called profit which is ticket sales minus film budget
df_movies$profit <- (df_movies$gross - df_movies$budget)

#Writing the CSV to a directory for use in Part 2 of the final project
write.csv(df_movies, 'C:\\Users\\PS3ma\\Documents\\Bellevue University\\DSC 630\\Cleaned_Movies.csv')

#Finding out which combination of genres had the highest IMDB score
imdbs <- df_movies %>% 
           group_by(genres) %>%
           summarise(across(imdb_score.x, mean))
imdbs[which.max(imdbs$imdb_score.x),]

mean(df_movies$imdb_score.x)
mean(df_movies$imdb_score.y)

