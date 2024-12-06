library("colorspace")

ROOT <- "C:/Users/jvavra/OneDrive - WU Wien/Documents/TBIP_colab/data/hein-daily/"
data <- read.csv(paste0(ROOT, 'ideal_data.csv'))
cols <- c(paste0("X", 0:24), 'avg', 'tbip')
R2 <- R2adj <- numeric(length(cols))
names(R2) <- names(R2adj) <- cols

for(y in cols){
  fit <- lm(as.formula(paste0(y, " ~ factor(party)")), data)
  sumfit <- summary(fit)
  R2[y] <- sumfit$r.squared
  R2adj[y] <- sumfit$adj.r.squared
}

ideal_data_R <- data.frame(R2 = R2,
                           R2adj = R2adj)
ideal_data_R
write.csv(ideal_data_R, paste0(ROOT, 'ideal_data_R.csv'))
