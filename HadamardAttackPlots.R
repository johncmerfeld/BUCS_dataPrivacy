setwd("Documents/Code/BUCS_dataPrivacy/")
library(ggplot2)
library(data.table)

results <- fread('HadamardResults.csv')
results$n <- as.factor(results$n)

ggplot(results,
       aes(x = `1/sigma`,
           y = pct_correct,
           color = n,
           group = n)) + 
  geom_line(alpha = 0.5,
            size = 2) + 
  scale_color_brewer(palette = "Spectral") +
  scale_x_log10() + 
  ggtitle("With low enough noise, even large datasets can be perfectly reconstructed") +
  xlab("1/\U03C3 (less noise \U2192)") + 
  ylab("ratio of secret bits correctly guessed") + 
  labs(color = "# of records")