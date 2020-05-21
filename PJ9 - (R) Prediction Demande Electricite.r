# # Github
# if(!require(devtools)) install.packages("devtools")
# devtools::install_github("sinhrks/ggfortify")
# install.packages('forecast', dependencies = TRUE)
# install.packages('caschrono', dependencies = TRUE)
# install.packages('readxl', dependencies = TRUE)

# -*- coding: utf8 -*-
library(dplyr) # très utile
library(reshape2) # dcast, melt
library(ggfortify) # autoplot
library(xts) # Séries temporelles
library(lubridate) # le choix dans la date
library(forecast) # Acf, Pacf, ma
library(tseries) # adf.test
library(caschrono) # t_stat
library(ggplot2)
library(readxl)
options(repr.plot.width=12, repr.plot.height=3)

getwd()
setwd("D:/DATA_ANALYST/WORK/PJ9/Data/R")

ARA0 = (read.csv("eco2mix-regional-cons-def_M.csv",sep=";",header=T)
        # Pour correspondre aux données DJU, on restreint la série sur une zone géographique
        %>% filter(Territoire=="Auvergne-Rhône-Alpes")
        # Seule les données dites définitives sont utilisées
        %>% filter(Qualité=="Données définitives")
        %>% select(c(Date, Consommation.totale))         
        %>% na.omit
        %>% setNames(c("Date","Conso")))
head(ARA0, 5)
# Conversion en série temporelle format xts
ARA0 = (xts(ARA0$Conso,order.by=as.Date(ARA0$Date,"%d.%m.%Y")) )# %>% apply.monthly(mean))
# Création d'un index base mensuelle
index(ARA0) <- floor_date(index(ARA0),"month")
# reconversion en timeserie
ARA0 = ts(ARA0,frequency=12,start=c(2013, 12), end=c(2018,12))
ARA0
# et voilà!
boxplot(ARA0 ~ cycle(ARA0),xlab="Mois",ylab="Consommation (KW/h)")

# Le datasource CEGIBAT permet uniquement d'exporter des données géolocalisées
# Plutôt que de devoir créer un utilitaire de requêtage puis consolidation (hors sujet)
# on concentre l'étude sur la région Nouvelle-Aquitaine.
# Autre option : acheter les DJU mensuels nationaux à Météo-France
DJU = (read.csv("dju_lyon_2012-2019.csv",sep=";",header=T)
        %>%  melt(id.vars = 'year')
        %>% na.omit
        %>% filter(value>=0)
        %>% filter(year>2013))
head(DJU, 5)
# Création d'un index base mensuelle
idx = floor_date(parse_date_time(paste(DJU$year,DJU$variable),"%Y %m"),"month")
# Conversion de la série en série temporelle format xts 
DJU = (xts(DJU$value,order.by=idx))
# reconversion en timeserie pour intégration avec Forecast
DJU = ts(DJU,frequency=12,start=c(2013, 12), end=c(2018,12))
DJU
# et voilà!
boxplot(DJU ~ cycle(DJU),xlab="Mois",ylab="DJU")

ARA0
DJU

# Regression linéaire des deux séries
# traitement particulier pour séries temporelles (c.f. doc lm)
base = ts.intersect(ARA0, DJU, dframe=TRUE) %>% na.omit
reg = lm(base, na.action=NULL)

plot(as.numeric(DJU),as.numeric(ARA0),xlab="DJU",ylab="ARA0",main="Régression linéaire de ARA0 par DJU")
abline(reg)
cbind("R2",round(summary(reg)$r.squared,2))

# On soustrait les résidus à la série
# res <- ts(residuals(reg),frequency=12,start=c(2013, 12), end=c(2018,12))
# ARA2 <- ARA0 - res
b = reg$coefficients[2]
effet_temp = ts(DJU*b,frequency=12,start=c(2013, 12), end=c(2018,12))
ARA2 = ARA0 - effet_temp

# Note : en pratique la régression linéaire n'est pas la méthode utilisée pour pondérer la consommation
# c.f. http://www.gpso-energie.fr/conseils/analyser-ses-consommations-dju 
# Consommations corrigées = consommations x (DJU de référence/DJU de la période de consommation considérée)

autoplot(cbind(ARA0,ARA2),xlab="Temps",ylab="Consommation (KW/h)")

autoplot(cbind(ARA0,ARA2,DJU),xlab="Temps",ylab="Consommation (KW/h)",facet=T)

options(repr.plot.height=7)
autoplot(decompose(ARA2),ylab="ARA2",main="")
options(repr.plot.height=3)

ARA2

options(warn = -1)
# Test de Dickey-Fuller
# Computes the Augmented Dickey-Fuller test for the null that x has a unit root.
adf.test(ARA2)
# H0 : ARA2 est stationnaire
# p-value inférieurs à 5%
options(warn = 0)

Acf(ARA2,main="Autocorrelation de la série ARA2 sur une période de 12 mois")

MA12 = ma(ARA2, order=12)

main="Moyenne mobile sur une période de 12 mois comme indiqué par le résultat ACF"
autoplot(cbind(ARA2,MA12),xlab="Temps",ylab="Consommation (KW/h)",main=main)

main="Détail de la série MA12 résultante"
autoplot(MA12,xlab="Temps",ylab="Consommation (KW/h)",main=main)

# From A Little Book of R For Time Series, Release 0.2
plotForecastErrors = function(forecasterrors) 
{
    forecasterrors = na.omit(forecasterrors)
    # make a histogram of the forecast errors:
    mybinsize = IQR(forecasterrors)/4
    mysd = sd(forecasterrors)
    mymin = min(forecasterrors) - mysd*5
    mymax = max(forecasterrors) + mysd*3
    # generate normally distributed data with mean 0 and standard deviation mysd
    mynorm = rnorm(10000, mean=0, sd=mysd)
    mymin2 = min(mynorm)
    mymax2 = max(mynorm)
    if (mymin2 < mymin) { mymin <- mymin2 }
    if (mymax2 > mymax) { mymax <- mymax2 }
    # make a red histogram of the forecast errors, with the normally distributed data overlaid:
    mybins = seq(mymin, mymax, mybinsize)
    hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
    # freq=FALSE ensures the area under the histogram = 1
    # generate normally distributed data with mean 0 and standard deviation mysd
    myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
    # plot the normal curve as a blue line on top of the histogram of forecast errors:
    points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

options(repr.plot.width=12)
HW  = forecast(HoltWinters(ARA0),12)
res = c(as.numeric(HW$fitted),as.numeric(HW$mean))
res = ts(res,frequency=12,start=c(2012, 12), end=c(2018,12))
α = HW$model$alpha
β = HW$model$beta
g = HW$model$gamma
main = sprintf("Estimation du modèle Holt-Winters (α:%.3f, β;%.3f, g:%.3f)\tRMSE: %.2f, MAPE: %.2f"
                ,α,β,g,accuracy(HW)[,"RMSE"],accuracy(HW)[,"MAPE"])
autoplot(cbind(ARA0,HoltWinters=res),xlab="Temps",ylab="Consommation (KW/h)", main=main)

options(repr.plot.width=4)
HW = HoltWinters(ARA0,
                  # alpha parameter of Holt-Winters Filter.
                  alpha = α,
                  # beta parameter of Holt-Winters Filter. 
                  # If set to FALSE, the function will do exponential smoothing.
                  beta = β,
                  # gamma parameter used for the seasonal component. 
                  # If set to FALSE, an non-seasonal model is fitted.
                  gamma = g,
                  #seasonal = "additive",
                  #start.periods = 12, l.start = NULL, b.start = NULL, s.start = NULL,
                  #optim.start = c(alpha = 0.3, beta = 0.1, gamma = 0.1),
                  #optim.control = list()
                  )
main = "Prédiction Holt-Winters sur un an"
res = forecast(HW,12)
res = autoplot(res,xlab="",ylab="",main=main)
res$coordinates$limits$x = c(2018,2020)
res
options(repr.plot.width=12)

# Partage de la série en training/testing
train = window(ARA0,end = c(2017,11))
test = window(ARA0,start = c(2017,12))
HoltWinters_ARA0 = forecast(HoltWinters(train, alpha=α, beta=β, gamma=g),12)

res = HoltWinters_ARA0
upper = ts(res$upper[,2],frequency=12,start=c(2017,12),end=c(2018,12))
lower = ts(res$lower[,2],frequency=12,start=c(2017,12),end=c(2018,12))
main="Modèle Holt-Winters : données de test et intervalle de confiance 95%"
autoplot(cbind(upper,lower,test),xlab="Temps",ylab="Consommation (KW/h)", main=main)

accuracy(HoltWinters_ARA0, test, d=12, D=1)

shapiro.test(HoltWinters_ARA0$residuals)
print("H0:normality")
Box.test(HoltWinters_ARA0$residuals, lag=12, type="Ljung-Box")
print("H0:independence")

options(repr.plot.width=4)
plotForecastErrors(HoltWinters_ARA0$residuals)
options(repr.plot.width=12)

options(repr.plot.width=4)
Acf(HoltWinters_ARA0$residuals)
options(repr.plot.width=12)

# Recommandations de la documentation pour trouver les paramètres optimaux
SARIMA = auto.arima(ARA0, stepwise=FALSE, approximation=FALSE, parallel=TRUE)
SARIMA

#Des résultats de auto.arima
order=c(0,0,1)
seasonal=c(1,1,1)

options(repr.plot.width=4)
Acf(ARA0,main="ARA0")
options(repr.plot.width=12)

options(repr.plot.width=4)
Pacf(ARA0,main="")
options(repr.plot.width=12)

options(repr.plot.width=4)
main = "Prédiction SARIMA sur un an"
res = forecast(SARIMA,12)
res = autoplot(res,xlab="",ylab="",main=main)
res$coordinates$limits$x <- c(2018,2020)
res
options(repr.plot.width=12)

res = forecast(SARIMA,12)
res = c(as.numeric(res$fitted),as.numeric(res$mean))
res = ts(res,frequency=12,start=c(2013,12),end=c(2018,12))
main = sprintf("Estimation du modèle SARIMA (1,0,0)(0,1,1)\tRMSE: %.2f, MAPE: %.2f",
                accuracy(SARIMA)[,"RMSE"],accuracy(SARIMA)[,"MAPE"])
autoplot(cbind(SARIMA=res,ARA0),xlab="Temps",ylab="Consommation (KW/h)", main=main)

# Partage de la série en training/testing
train = window(ARA0,end = c(2017,11))
test = window(ARA0,start = c(2017,12))
SARIMA = forecast(Arima(train,order=order,seasonal=seasonal),12)

res = SARIMA
upper = ts(res$upper[,2],frequency=12,start=c(2017,12),end=c(2018,12))
lower = ts(res$lower[,2],frequency=12,start=c(2017,12),end=c(2018,12))
main="Modèle SARIMA : données de test et intervalle de confiance 95%"
autoplot(cbind(upper,lower,test),xlab="Temps",ylab="Consommation (KW/h)", main=main)

accuracy(SARIMA, test)

shapiro.test(SARIMA$residuals)
print("H0:normality")
Box.test(SARIMA$residuals, fitdf=1+0, lag=12, type="Ljung-Box")
print("H0:independence")

options(repr.plot.width=4)
plotForecastErrors(SARIMA$residuals)
options(repr.plot.width=12)

options(repr.plot.width=4)
Acf(SARIMA$residuals)
options(repr.plot.width=12)


