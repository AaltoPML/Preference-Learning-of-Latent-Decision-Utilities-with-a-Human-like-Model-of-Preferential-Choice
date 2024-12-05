download.file("https://github.com/aaronrkaufman/compactness/blob/master/data/preds.RData", "preds.RData", "auto")
datafile = "preds.RData"
load(datafile)
df = as.data.frame(finalpreds)
write.csv(df, "feature_data.csv", row.names=FALSE)