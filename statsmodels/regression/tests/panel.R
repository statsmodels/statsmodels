source('../../tools/R2nparray/R/R2nparray.R')

library(plm)
dat = read.csv('../../datasets/grunfeld/grunfeld.csv')
dat = pdata.frame(dat, c('firm', 'year'))
f = invest ~ value + capital

models = list(
    "within" = plm(f, data=dat, model='within'),
    "swar1w" = plm(f, data=dat, model='random'),
    "pooling" = plm(f, data=dat, model='pooling'),
    "between" = plm(f, data=dat, model='between')
    )

black = c('coefficients', 'predict', 'model')
for (m in names(models)){
    models[[m]] = c(models[[m]], apply_functions(models[[m]]))
    prefix = paste(m, ".", sep='')
    models[[m]]$fittedvalues = as.numeric(models[[m]]$fittedvalues)
    models[[m]] = convert_items(models[[m]], prefix=prefix, blacklist=black)
}
 
sink('results_panel_pandas.py')
write_header()
cat('\n')
# Insert test class for each model
for (m in names(models)) {
    cat(paste('\n\n#########', m, '\n',  m, ' = Bunch()\n', sep=''))
    cat(models[[m]], labels='    ')
}
sink()
