setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)

results <- fread('RandomQueryResults.csv')
results$nmgroup = as.factor(results$n + results$m_factor)

ggplot(results,
       aes(x = `1/sigma`,
           y = pct_correct,
           color = as.factor(n),
           group = nmgroup)) + 
  geom_line(alpha = 0.5,
            aes(size = m_factor)) + 
  scale_color_brewer(palette = "Dark2") +
  scale_x_log10() + 
  ggtitle("Larger random query matrices can reconstruct noisier datasets") +
  xlab("1/\U03C3 (less noise \U2192)") + 
  ylab("ratio of secret bits correctly guessed") + 
  labs(color = "# of records",
       size = "ratio of queries to records")