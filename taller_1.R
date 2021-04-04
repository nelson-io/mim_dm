# set libs
library(tidyverse)
library(ISLR)
library(rio)
library(rpart)
library(caret)
library(rpart.plot)

#import data

anames <- c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'pred')
adata <- import('data/adult_data.txt') %>% set_names(anames) %>% 
  mutate(pred = as.factor(if_else(pred == '<=50K',0,1)))

atest <- import('data/adult_test.txt') %>% set_names(anames) %>% 
  mutate(pred = as.factor(if_else(pred == '<=50K.',0,1)))
 
#decision tree
#max depth
#minsplit 1
#minbucket 1
# multiple cp with k-fold


set.seed(69)
# set control

rpartcontrol <- rpart.control(minsplit = 1, minbucket = 1, cp = seq(.1,1,.1))

tree_fit <- rpart(pred~.,
                  data = adata,
                  control = rpartcontrol)

tree_predictions <- predict(tree_fit, newdata = atest %>% select(-pred), type = "class")

print(mean(atest$pred == tree_predictions)) # acc = 84.45%

#k-fold cp optimization

k_fold_part <- function(df, k){
  
  x <- df %>% 
    sample_n(nrow(.)) %>% 
    mutate(rown = 1:nrow(.),
           fold = cut(rown, k, labels = 1:k)) %>% 
    select(-rown)
  
  return(x)
}

folds_df <- k_fold_part(adata,5) %>% 
  filter(native_country != 'Holand-Netherlands')

ev_df <- data.frame()

for(i in seq(0,1,length.out = 100)){
  k <- c()
  acc <- c()
  for(j in 1:5){
    val_data <- folds_df %>% filter(fold == j)
    t_data <- folds_df %>% filter(!fold == j)
    
    
    tree_fit <- rpart(pred~. ,
                      data = t_data,
                      control = rpart.control(minsplit = 1, minbucket = 1, cp = i))
    
    tree_predictions <- predict(tree_fit, newdata = val_data %>% select(-pred), type = "class")
    
    acc <- c(acc, mean(val_data$pred == tree_predictions))
    
  }
  
  ev_df <- rbind(ev_df, data.frame(k = i, acc= mean(acc)))
  
}

#best cv model
ev_df %>% slice(which.max(ev_df$acc))


# Entreno la mejor configuración con todos los datos
best_tree <- rpart(pred ~ .,
                   data = adata,
                   control = rpart.control(minsplit = 1,
                                           minbucket = 1,
                                           cp=0.01010101))

rpart.plot(best_tree)


best_tree_predictions <- predict(best_tree, newdata = atest %>% select(-pred), type = "class")

mean(atest$pred == best_tree_predictions)

#La performance fue muy similar

# ¿Es cierto que cuando uno hace overfitting tanto el error en entrenamiento como el validación 
# suelen ser altos? Justifique su respuesta.

#No es cierto, ya que disminuye el error en entrenamiento pero aumenta en validación ya que el modelo se ajusta a los puntos
# y la alta flexibilidad hace que sea malo para predecir nueva información.

# ¿Cuál es el objetivo final de usar técnicas como la de holdout set?
# el objetivo es optimizar la selección del modelo e hiperparams pudiendo validar métricas de error que soporten la decisión a tomar
# para luego usar esa configuración para entrenar al modelo final

# ¿Es cierto que tanto en holdoutset como en k-fold crossvalidation al reordenar losdatos de manera aleatoria se garantiza
# que uno va a tener una buena estimación de la performance del modelo en datos desconocidos? Justifique su respuesta.

#no necesariamente. se debe ajustar el método de validación más propicio de acuerdo al tipo de datos. Con muy pocos datos (LOOCV) no es lo mismo que 
# con muchos o si estoy manejando series de tiempo.

# ¿Puede darse la situación en que el valor de minbucket afecte cómo el árbol 
# parte un nodo particular pero el valor de minsplit no?  Piense y explique un ejemplo concreto.

#Puede darse el caso. Un ejemplo lo constituye una partición en una rama cuya cantidad de observaciones admite una 
# nueva partición ya que es mayor a minsplit, pero la asimetría de los nodos resultantes hace que alguno de ellos no cuente con 
# más observaciones que las definidas en el parámetro minbucket.

# ¿Es cierto que árboles de decisión es un modelo lazy? Justifique su respuesta.

# no es cierto, DT es un modelo Eager ya que el modelo calcula todos los puntos de corte y al predecir una nueva observación simplemente
#  se clasifica con las reglas predefinidas, sin necesidad de reevaluar el modelo.
