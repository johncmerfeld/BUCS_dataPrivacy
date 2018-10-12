setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)

results <- fread('RunningCounter.csv')

ggplot(results,
       aes(x = n,
           y = pct_correct)) +
  geom_point(alpha = 0.4,
             color = "darkgreen") +
  ggtitle("The 'naive' algorithm converges to 75% of users correctly identified") +
  xlab("number of users in set") + 
  ylab("ratio of users correctly guessed") +
  ylim(0.7, 0.9)

results <- fread('RunningCounterExtra.csv')

ggplot(results,
       aes(x = n,
           y = pct_correct)) +
  geom_point(alpha = 0.4,
             color = "purple") +
  ggtitle("The 'extra' algorithm converges to 83% of users correctly identified") +
  xlab("number of users in set") + 
  ylab("ratio of users correctly guessed") +
  ylim(0.7, 0.9)