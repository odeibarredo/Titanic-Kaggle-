######   KAGGLE - TITANIC    #####

### LIBRARY NEEDED
library(ggplot2)
library(gridExtra)
library(psych)
library(caret)
library(plyr)
library(randomForest)
library(rpart)
library(e1071)
library(kernlab)

### LOAD DATA
original_train <- read.csv("train.csv", sep=";", header=TRUE)
original_test <- read.csv("test.csv", sep=";", header=TRUE)

# Combine both datasets for the later processing of data
original_test$survived <- NA                    
full <- rbind(original_train, original_test)

### DATA MINING
# Quick check of data
summary(full)
str(full)
prop.table(table(original_train$survived)) # ~ 61% of people died

# Transform "survived" and "class" to factors"
full$survived <- as.factor(full$survived)
full$pclass <- as.factor(full$pclass)
original_train$survived <- as.factor(original_train$survived)
original_train$pclass <- as.factor(original_train$pclass)
original_test$pclass <- as.factor(original_test$pclass)

# "Women and children first" & "Money rules the world"
# Let's see how sex, age and pclass influence on the passengers fate

# Variables overview
prop.table(table(original_train$sex)) # ~ 35% female | 65% male
prop.table(table(original_train$pclass)) # ~ 24% 1st class | 21% 2nd | 55% 3rd
ggplot(original_train, aes(x=age)) +
  geom_histogram() + xlab("Age") + ylab("Total count") 
# most passengers between 20-40 years old

ggplot(original_train, aes(x=age, fill=survived)) +
  geom_histogram() + xlab("Age") + ylab("Total count") +
  facet_wrap(~ pclass + sex, nrow = 3) 

# Pclass: lower the social class lower chance of surviving
# Sex: women had better chance of surviving
# Age: overall kids had better chance of surviving and the older the 
#      darker the fate
# Most deaths were from 3rd class male young people, followed by
# young male people from 2nd class

# Now let's check the other variables
# Family:

sibsp_plot <- ggplot(original_train, aes(x=sibsp, fill = survived)) +
  geom_bar() + ylim(0,700) + xlab("Number of siblings/spouses") + 
  ylab("Total count") + 
  theme(legend.position=c(0.9,0.9), 
        legend.background = element_rect(fill = "grey90", color="black",
                                         linetype = "solid"))
  
parch_plot <- ggplot(original_train, aes(x=parch, fill = survived)) +
  geom_bar() + ylim(0,700) + xlab("Number of parents/children") +
  theme(axis.title.y=element_blank()) +
  theme(legend.position="none")

grid.arrange(sibsp_plot, parch_plot, nrow=1)

# Most people were travelling alone, and had a ~ 45% survival rate
# In the other hand, you had better surviving chances if you 
# had 1 or 2 family members, and worse the bigger the family members

# Combining this two variables into one could be useful in the prediction
full$family_size <- full$sibsp + full$parch

# Travelling variables:
# Embarking:
# This variable has 2 empty cells that we need to change into NA (for now)
table(original_train$embarked)
levels(original_train$embarked)[1] <- NA

embarking_plot <- ggplot(original_train[!is.na(original_train$embarked),],  
                  aes(x=embarked, fill = survived)) +
                  geom_bar() + xlab("Embark") + ylab("Total Count") +
                  theme(legend.position=c(0.1,0.9), 
                  legend.background = element_rect(fill = "grey90", 
                                                   color="black",
                                                   linetype = "solid"))

# Most passenger were from Southampton, where the survival chance was
# the lowest

# Fare:
head(table(original_train$fare)) # there are no empty cells
summary(original_train$fare)
fare_plot <- ggplot(original_train, aes(x=fare, fill = survived)) +
  geom_histogram() + xlab("Fare") + ylab("Total count") +
  theme(legend.position="none")

# Following the social class logic, people who paid more for the cruise
# had better survival rate

# Cabin:
head(table(original_train$cabin)) # there are 687 empty cells
# That's a lot of missing information, which suggest that we may have to
# discard this variable, unless we find that it has any correlation
# with survival rate

# Cabins seem to be all starting with a letter followed by a number
# For a better exploration we'll take that first character and create a new
# variable with it
cabin_data <- as.character(original_train$cabin)
cabin_first_char <- as.factor(substr(cabin_data, 1, 1))
original_train$cabin_f_ch <- cabin_first_char

levels(original_train$cabin_f_ch)[1] <- NA
table(original_train$cabin_f_ch)

ggplot(original_train, aes(x=cabin_f_ch, fill=survived)) +
  geom_bar() +  xlab("First char") + ylab("Total count") +
  facet_wrap(~pclass)

# most NA are from 3rd class passegers

cabin_plot <- ggplot(original_train[!is.na(original_train$cabin_f_ch),], 
              aes(x=cabin_f_ch, fill=survived)) +
              geom_bar() +  xlab("Cabin first char") + ylab("Total count") +
              theme(legend.position="none")


# There seems to be some slight differences between cabins, let's
# check it out
pairs.panels(original_train[c("survived", "cabin_f_ch", "pclass")])

# Actually no, there is correlation between cabins and pclass (as commented
# before), but not between cabins and survival

# So, in recap, the cabin variable doesn't seem to be a useful variable
# for the survival prediction

# Ticket:
head(table(original_train$ticket)) # there are no empty cells
# We'll process tickets the same way as cabins, taking the first character

ticket_data <- as.character(original_train$ticket)
ticket_first_char <- as.factor(substr(ticket_data, 1, 1))
original_train$ticket_f_ch <- ticket_first_char

table(original_train$ticket_f_ch)

ticket_plot <- ggplot(original_train, aes(x=ticket_f_ch, fill=survived)) +
               geom_bar() +  xlab("Ticket firts char") +
               ylab("Total count") +
               theme(legend.position="none")

# First three digits are the most common tickets, followed by P, S, C and A

ggplot(original_train, aes(x=ticket_f_ch, fill=survived)) +
  geom_bar() +  xlab("First char") + ylab("Total count") +
  facet_wrap(~pclass)

# The first three correlate with the social class
# P appears more in 1st class
# S appears more in 2nd and 3rd class
# C and A appear more in 3rd class

# Traveling plots:
grid.arrange(embarking_plot, fare_plot, cabin_plot, ticket_plot)

# Correlation between all variables:
pairs.panels(original_train[c("survived", "pclass", "sex", "age", "sibsp",
                              "parch", "embarked", "fare", "cabin_f_ch",
                              "ticket_f_ch")])

# Ok, so now that we have check out all the variables is time to 
# start training our predictive modeling
# The first thing we need to do is fill in the missing values based on
# the available data

### NA's TREATMENT
# We've seen NA's in embarked and fare, but there are also many empty cells
# in age

# Embarked:
table(full$embarked)
levels(full$embarked)[1] <- NA

full[is.na(full$embarked),]
# There are only 2 NA's, both from the 1st class, females and survivals

ggplot(full[full$sex=="female" & full$pclass=="2", ],
       aes(x=embarked, fill = survived)) +
       geom_bar() + ylab("Total Count") 

# Based on the info the best choice is to fill those NA with the most 
# common value, in this case S
full$embarked[is.na(full$embarked)] <- "S"

# Fare:
full[which(is.na(full$fare)), ]
# One male, 3rd class, 60 years,...

p3_m_fare <- na.omit(full$fare[full$sex=="male" & full$pclass=="3"])
summary(p3_m_fare)
# We'll take the median
full$fare[is.na(full$fare)] <- 7.9

# Age:
summary(full$age) # there are 263 NA's
# Too many cases to replace them with the most common value or to go one
# by one exploring each case

# For each person we could look at his social class first, then use the
# corresponding median age of that class to replace the NA. 
# But there is actually a better way, and that is using the
# persons title (wich is included in it's name)

## Create 'Title' variable
names <- as.character(full$name)

extract_title <- function(x){
  extr <- strsplit(x, split = "[,.]")[[1]][2]
}
titles <- sapply(names, extract_title)
titles <- gsub(" ", "", titles) # demove the blank space at the beginning
table(titles)

full$title <- as.factor(titles)

# Get each title mean age
titles_age <- ddply(full,~na.omit(title),summarise,
                    mean=round(mean(na.omit(age)),2))
colnames(titles_age)[1] <- "Title"

# Replace age NA's:
age_na <- data.frame("Name"= full$name[is.na(full$age)], 
                      "Title"= full$title[is.na(full$age)],
                      "Age"= NA)

for(i in 1:nrow(age_na)){
  if (age_na$Title[i] %in% levels(full$title)){
    age_na$Age[i] <- titles_age$mean[match(age_na$Title[i], levels(full$title))]
  }
}

for(i in 1:nrow(full)){
  if(is.na(full$age[i])){
    full$age[i] <- age_na$Age[match(full$name[i], age_na$Name)]
  }
}

# Lastly, we have to create the new variable with the ticket first 
# character in the full dataset

full_ticket <- as.character(full$ticket)
full_ticket_char <- as.factor(substr(full_ticket, 1, 1))
full$ticket_f_ch <- full_ticket_char


#### MACHINE LEARNING
# Data
train <- full[1:891,]
test <-  full[892:1309, ]

# Split the training set
set.seed(1)
trainIndex <- sample(1:nrow(train), size = 0.8*nrow(train))
training <- train[trainIndex, ]
testing <- train[-trainIndex, ]

# Random forest
rf_1 <- randomForest(survived ~ sex + age + pclass + sibsp + parch +
                      embarked + fare + ticket_f_ch, data = training, 
                     importance = TRUE, ntree = 2000)

pred_1 <- predict(rf_1, testing)
confusionMatrix(testing$survived, pred_1)
# First try: 83% Accuracy. Not bad!
# Check which variables are the most influencial in the predictive model
varImpPlot(rf_1,type=2)

# What if we include the title varible?
rf_2 <- randomForest(survived ~ sex + age + pclass + sibsp + parch +
                       embarked + fare + ticket_f_ch + title, 
                     data = training, 
                     importance = TRUE, ntree = 2000)

pred_2 <- predict(rf_2, testing)
confusionMatrix(testing$survived, pred_2)
# ~ %85 accuracy, a little better


# what about the using the family_size variable
rf_3 <- randomForest(survived ~ sex + age + pclass + family_size +
                     embarked + fare + ticket_f_ch + title,
                     data = training, 
                     importance = TRUE, ntree = 2000)

pred_3 <- predict(rf_3, testing)
confusionMatrix(testing$survived, pred_3)
# not much change...

# What about scaling numeric variables?
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

train_numeric <- train[,c("age", "fare")]
train_norm <- as.data.frame(lapply(train_numeric, normalize))
train <- cbind(train, age_n=train_norm$age,
                    fare_n=train_norm$fare)

n_train <- train[trainIndex, ]
n_test <- train[-trainIndex, ]

rf_norm <- randomForest(survived ~ sex + age_n + pclass + sibsp +
                       embarked + fare_n + ticket_f_ch + title, 
                       data = n_train, 
                       importance = TRUE, ntree = 2000)

pred_norm <- predict(rf_norm, n_test)
confusionMatrix(n_test$survived, pred_norm)
# not much of a difference


# Other predictive models?

# Decision tree:
dt_model <- rpart(survived  ~ sex + age + pclass + sibsp +
                  embarked + fare + ticket_f_ch + title, 
                  data = training,
                  method = "class")
dt_pred <- predict(dt_model, testing, type = "class")
confusionMatrix(testing$survived, dt_pred)
# worse than random forest...

# Naive bayes:
nb_model <- naiveBayes(survived  ~ sex + age + pclass + sibsp +
                       embarked + fare + ticket_f_ch + title, 
                       data = training,
                       method = "class")
nb_pred <- predict(nb_model, testing)
confusionMatrix(testing$survived, nb_pred)
# even worse...

# SVM:
svm_model <- train(survived  ~ sex + age + pclass + sibsp +
                   embarked + fare + ticket_f_ch + title,
                   data = training,
                   method="svmRadial")
svm_pred <- predict(svm_model, testing)

confusionMatrix(testing$survived, svm_pred)
# not better...

# Recap: best model seems to be the random forest
random_model <- randomForest(survived ~ sex + age + pclass + sibsp + 
                               parch + embarked + fare + ticket_f_ch + 
                               title, 
                               data = train, 
                               importance = TRUE, ntree = 2000)

random_pred <- predict(random_model, test)

# Submission
submit_1 <- data.frame(PassengerId = rep(892:1309), Survived = random_pred)

write.csv(submit_1, file = "Titanic_subm_1.csv", row.names = FALSE)

# Final note:
# It's your turn to play with the data and find best prediction model

# THE END!

