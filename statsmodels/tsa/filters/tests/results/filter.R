library(R2nparray)

x <- c(-50, 175, 149, 214, 247, 237, 225, 329, 729, 809, 530, 489, 540, 457, 
         195, 176, 337, 239, 128, 102, 232, 429, 3, 98, 43, -141, -77, -13, 
         125, 361, -45, 184)
x <- ts(x, start=c(1951, 1), frequency=4)


conv2 <- filter(x, c(.75, .25), method="convolution")

conv1 <- filter(x, c(.75, .25), method="convolution", sides=1)

recurse <- filter(x, c(.75, .25), method="recursive")

recurse.init <- filter(x, c(.75, .25), method="recursive", init=c(150, 100))

conv2.odd <- filter(x, c(.75, .5, .3, .2, .1), method="convolution", 
                     sides=2)
conv1.odd <- filter(x, c(.75, .5, .3, .2, .1), method="convolution", 
                     sides=1)
recurse.odd <- filter(x, c(.75, .5, .3, .2, .1), method="recursive", 
                       init=c(150, 100, 125, 135, 145))

# missing values

x[10] = NaN

conv2.na <- filter(x, c(.75, .25), method="convolution")

conv1.na <- filter(x, c(.75, .25), method="convolution", sides=1)

recurse.na <- filter(x, c(.75, .25), method="recursive")

recurse.init.na <- filter(x, c(.75, .25), method="recursive", init=c(150, 100))


options(digits=12)

R2nparray(list(conv2=as.numeric(conv2), conv1=as.numeric(conv1), 
               recurse=as.numeric(recurse), 
               recurse_init=as.numeric(recurse.init),
               conv2_na=as.numeric(conv2.na), conv1_na=as.numeric(conv1.na), 
               recurse_na=as.numeric(recurse.na), 
               recurse_init_na=as.numeric(recurse.init.na),
               conv2_odd=as.numeric(conv2.odd),
               conv1_odd=as.numeric(conv1.odd),
               recurse_odd=as.numeric(recurse.odd)),
          fname="filter_results.py")

