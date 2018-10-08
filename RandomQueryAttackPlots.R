setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)

results <- fread('RandomQueryResults.csv')
results$nmgroup = as.factor(results$n * results$m)

ggplot(results,
       aes(x = `1/sigma`,
           y = pct_correct,
           color = as.factor(n),
           group = nmgroup)) + 
  geom_line(alpha = 0.5,
            aes(size = m)) + 
  scale_x_log10() + 
  ggtitle("With low enough noise, even large datasets can be perfectly reconstructed") +
  xlab("less noise \U2192") + 
  ylab("ratio of secret bits correctly guessed")