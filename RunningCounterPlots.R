setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)

results <- fread('RunningCounter.csv')
results$algorithm <- 'naive'
results2 <- fread('RunningCounterExtra.csv')
results2$algorithm <- 'extra'

results <- rbind(results, results2)

ggplot(results,
       aes(x = n,
           y = pct_correct,
           color = algorithm)) +
  geom_point(alpha = 0.8) +
  ggtitle("The 'naive' algorithm converges to 75% of users correctly identified \n The 'extra' algorithm converges to 83.3%") +
  xlab("number of users in set") + 
  ylab("ratio of users correctly guessed") +
  scale_color_brewer(palette = "Accent") 
  ylim(0.7, 0.9)
