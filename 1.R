library(caret)
library(vegan)
library(ggplot2)
library(mlbench)
library(MLeval)

# runtime vars
data(PimaIndiansDiabetes, package = "mlbench") ; df = PimaIndiansDiabetes ; rm("PimaIndiansDiabetes")
#df = iris
#data("BreastCancer") ; df = BreastCancer ; rm("BreastCancer")
set.seed(123)
seed_n = 4
loop_n = 2
train_frac = 0.7
class_col = ncol(df) # always last


# abalone_no_nzv_pca <- preProcess(select(abalone_train, - old), 
#                                  method = c("center", "scale", "nzv", "pca"))
# 

##### 1 - test:train split
train_i = createDataPartition(df[,class_col],
                                   p = train_frac,
                                   list = FALSE,
                                   times = 1)
train_n = length(train_i)
train = df[train_i, ]
test = df[-train_i, ]
rownames(train) = 1:nrow(train)
rownames(test) = 1:nrow(test)


##### 2 - initialse a random seed set of samples
which_tested = c()
which_tested = c(which_tested, sample(1:nrow(train), seed_n)) ; which_tested_random = which_tested ; which_tested_uncertainty = which_tested

# plot
#mds = data.frame(vegan::metaMDS(train[,1:8],k = 2,trymax = 10)$points)
mds = data.frame(prcomp(train[,1:(class_col - 1)], center = TRUE,scale. = TRUE)$x)

# plot and save full
g = ggplot(mds, aes(x = PC1, y = PC2, colour = train[,class_col]) ) +
       geom_point()
ggsave("0_all_labs_pc12.png", plot = g, device = "png",width = 10, height = 10)
g = ggplot(mds, aes(x = PC1, y = PC3, colour = train[,class_col]) ) +
  geom_point()
ggsave("0_all_labs_pc13.png", plot = g, device = "png",width = 10, height = 10)
g = ggplot(mds, aes(x = PC2, y = PC3, colour = train[,class_col]) ) +
  geom_point()
ggsave("0_all_labs_pc23.png", plot = g, device = "png",width = 10, height = 10)


# plot the current known
ggplot(mds[which_tested,], aes(x = PC1, y = PC2, colour = train[which_tested,class_col] )) +
  geom_point() +
  geom_point(data = mds[-which_tested,], colour = "grey" ,alpha = 0.1) +
  theme_classic() +
  labs(colour = "class")
ggsave("z_init.png", device = "png")

# so now train a classifier, how well does it do?
fit_control = trainControl(method = "cv", 
                                 number = 1, 
                                 savePredictions = TRUE, 
                                 classProbs = TRUE, 
                                 verboseIter = TRUE)
rf_fit = suppressWarnings({
  train(y = as.factor(train[which_tested, class_col]), 
                x = train[which_tested,1:(class_col - 1)], 
                na.action  = na.pass,
                method = "rf",
               control = fit_control)
  })
                
model_test_prob = predict(rf_fit, test[,1:(class_col - 1)], type =  "prob")
# extra fluff is just to suppress figures
t = MLeval::evalm(data.frame(model_test_prob, test[,class_col]) , plots = F, silent = T)
AUC_ROC_random = t$stdres$Group1$Score[13]
AUC_ROC_uncertainty = AUC_ROC_random



##### 3 = loop iterations
# now next batch
# which most uncertain?
for(i in 1:10){
  model_test_prob = predict(rf_fit, train[-which_tested,], type =  "prob")
  model_test_prob$index = rownames(model_test_prob)
  model_test_prob$uncertainty = 1 - abs( 0.5 - model_test_prob$pos)
  model_test_prob = model_test_prob[order(-model_test_prob$uncertainty),]
  which_next_batch_uncertainty = as.numeric(model_test_prob$index[1:loop_n])
  which_next_batch_random = as.numeric(sample(model_test_prob$index, loop_n))
  
  # now iterate and store ROC for random and uncertainty
  which_tested_random = c(which_tested_random, which_next_batch_random)
  rf_fit = train(y = as.factor(train[which_tested_random, class_col]), 
                 x = train[which_tested_random,1:(class_col - 1)], 
                 na.action  = na.pass,
                 method = "rf",
                 control = fit_control)
  model_test_prob = predict(rf_fit, test[,1:(class_col - 1)], type =  "prob")
  t = MLeval::evalm(data.frame(model_test_prob, test[,class_col]) , plots = F, silent = T)
  AUC_ROC_random = c(AUC_ROC_random, t$stdres$Group1$Score[13])
  
  
  which_tested_uncertainty = c(which_tested_uncertainty, which_next_batch_uncertainty)
  rf_fit = train(y = as.factor(train[which_tested_uncertainty, class_col]), 
                 x = train[which_tested_uncertainty,1:(class_col - 1)], 
                 na.action  = na.pass,
                 method = "rf",
                 control = fit_control)
  model_test_prob = predict(rf_fit, test[,1:(class_col - 1)], type =  "prob")
  t = MLeval::evalm(data.frame(model_test_prob, test[,class_col]) , plots = F, silent = T)
  AUC_ROC_uncertainty = c(AUC_ROC_uncertainty, t$stdres$Group1$Score[13])
  
  # plot the current known
  ggplot(mds[which_tested_uncertainty,], aes(x = PC1, y = PC2, colour = train[which_tested_uncertainty,class_col] )) +
    geom_point() +
    geom_point(data = mds[-which_tested_uncertainty,], colour = "grey" ,alpha = 0.1) +
    theme_classic() +
    labs(colour = "class", subtitle = "")
  ggsave(paste0("z_uncert_",i,".png"),device = "png")
  # plot the current known
  ggplot(mds[which_tested_random,], aes(x = PC1, y = PC2, colour = train[which_tested_random,class_col] )) +
    geom_point() +
    geom_point(data = mds[-which_tested_random,], colour = "grey" ,alpha = 0.1) +
    theme_classic() +
    labs(colour = "class")
  ggsave(paste0("z_rand_",i,".png"),device = "png")
  
  print(paste(AUC_ROC_random[length(AUC_ROC_random)],
        AUC_ROC_uncertainty[length(AUC_ROC_uncertainty)]))
}

ggplot()