#libs
library(tidyverse)
library(e1071)
library(mlbench)
library(class)
library(ISLR)
library(kernlab)

#import data
df <- read_delim("data/bankruptcy_data_red.txt",delim = '\t')




#Ensure replicability
set.seed(0)

#make train and test sets
test_vect <- sample(1:nrow(df),size = 300)

train_df <- df[-test_vect,]
test_df <- df[test_vect,]


#train naive Bayes

nb_classifier <- naiveBayes(class ~ ., data = train_df)

# Train  accuracy
print(mean(predict(nb_classifier, newdata = train_df) == train_df$class))

# Test accuracy
print(mean(predict(nb_classifier, newdata = test_df) == test_df$class))


#add-1 smoothing makes sense?

# no, continuous variables

#knn no scaling


train_acc <- map_dbl(1:100,~ mean(train_df$class == knn(train = train_df[,1:ncol(df)-1],
                                                        test = train_df[,1:ncol(df)-1], 
                                                        cl = train_df$class, 
                                                        k=.x)))

test_acc <- map_dbl(1:100,~ mean(test_df$class == knn(train = train_df[,1:ncol(df)-1],
                                                        test = test_df[,1:ncol(df)-1], 
                                                        cl = train_df$class, 
                                                        k=.x)))

knn_no_scaling_df <- data.frame(
  k = 1:100,
  train_acc = train_acc,
  test_acc = test_acc
)


ggplot(knn_no_scaling_df, aes(x = k))+
  geom_line(aes(y = train_acc), color = 'red')+
  geom_line(aes(y = test_acc), color = 'steelblue')+
  geom_point(x=knn_no_scaling_df$k[which.max(knn_no_scaling_df$test_acc)],
             y=max(knn_no_scaling_df$test_acc), size = 3, color = 'darkgreen')+
  ylab('train and test accuracy')+
  ggtitle(paste0('Best test performance at k = ',
                 knn_no_scaling_df$k[which.max(knn_no_scaling_df$test_acc)]))




#knn  scaling
#scale

df_scaled <- data.frame(cbind(
  scale(df[,1:ncol(df) -1]),
  df$class)) %>% set_names(names(df))

train_df <- df_scaled[-test_vect,]
test_df <- df_scaled[test_vect,]


train_acc <- map_dbl(1:100,~ mean(train_df$class == knn(train = train_df[,1:ncol(df)-1],
                                                        test = train_df[,1:ncol(df)-1], 
                                                        cl = train_df$class, 
                                                        k=.x)))

test_acc <- map_dbl(1:100,~ mean(test_df$class == knn(train = train_df[,1:ncol(df)-1],
                                                      test = test_df[,1:ncol(df)-1], 
                                                      cl = train_df$class, 
                                                      k=.x)))

knn_scaled_df <- data.frame(
  k = 1:100,
  train_acc = train_acc,
  test_acc = test_acc
)


ggplot(knn_scaled_df, aes(x = k))+
  geom_line(aes(y = train_acc), color = 'red')+
  geom_line(aes(y = test_acc), color = 'steelblue')+
  geom_point(x=knn_scaled_df$k[which.max(knn_scaled_df$test_acc)],
             y=max(knn_scaled_df$test_acc), size = 3, color = 'darkgreen')+
  ylab('train and test accuracy')+
  ggtitle(paste0('Best test performance at k = ',
                 knn_scaled_df$k[which.max(knn_scaled_df$test_acc)]))




# best test performance at k=1 scaled, scaling improves model and setting a seed ensures reproducibility


#¿Si en Bayes ingenuo al usar suavizado aditivo se aumenta el valor de α haciendo que tienda a infinito,
#a qué valor tiende P(Ck|xi)? Justifique su respuesta.

#tiende a 1/K. las ocurrencias dejan de ser relevantes e importa sólo la cantidad de clases.



# ¿Si se usa vecinos cercanos para regresión, a medida que aumenta el número de vecinos  a considerar 
# por el algoritmo, a qué valor tiende la predicción? Justifique su respuesta.

# Se tiende al promedio, dado que es el valor que se obtendría al contemplar la totalidad de los datos

# ¿Por qué Bayes ingenuo no es bueno captando interacciones complejas entre variables?
# Justifique su respuesta.

# Sólo contempla priors y probabilidades condicionadas asumiendo que no hay correlación entre estas.
#  de esta manera, cualquier tipo de interacción multivariada queda por fuera del alcance del algoritmo

# ¿Si usted tuviera la necesidad de, una vez entrenado el modelo, predecir muy rápido sobre nuevas 
# observaciones, elegiría el modelo de bayes ingenuo o el de vecinos más cercanos? 
# Justifique su respuesta.

#Usaría Naive Bayes ya que es Eager por lo que el costo computacional de predecir una nueva 
# nueva observación es considerablemente menor.
