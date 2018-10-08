setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)
library(sqldf)

results <- fread('HadamardResults.csv')
results$n <- as.factor(results$n)

ggplot(results,
       aes(x = `1/sigma`,
           y = pct_correct,
           color = n,
           group = n)) + 
  geom_line(alpha = 0.5,
            size = 2) + 
  scale_x_log10() + 
  ggtitle("With low enough noise, even large datasets can be perfectly reconstructed") +
  xlab("less noise \U2192") + 
  ylab("ratio of secret bits correctly guessed")