download.file("https://github.com/aaronrkaufman/compactness/blob/master/data/training_data.RData", "training_data.RData", "auto")
datafile = "training_data.RData"
load(datafile)
df = do.call(rbind, mylist)
fold_id <- rownames(df)
write.csv(cbind(fold_id, df), "ranking_data.csv", row.names=FALSE)