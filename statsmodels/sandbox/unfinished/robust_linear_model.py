if __name__=="__main__":
#NOTE: This is to be removed
#Delivery Time Data is taken from Montgomery and Peck
    import statsmodels.api as sm

#delivery time(minutes)
    endog = np.array([16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.00, 17.83,
    79.24, 21.50, 40.33, 21.00, 13.50, 19.75, 24.00, 29.00, 15.35, 19.00,
    9.50, 35.10, 17.90, 52.32, 18.75, 19.83, 10.75])

#number of cases, distance (Feet)
    exog = np.array([[7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6,
    7, 3, 17, 10, 26, 9, 8, 4], [560, 220, 340, 80, 150, 330, 110, 210, 1460,
    605, 688, 215, 255, 462, 448, 776, 200, 132, 36, 770, 140, 810, 450, 635,
    150]])
    exog = exog.T
    exog = sm.add_constant(exog)

#    model_ols = models.regression.OLS(endog, exog)
#    results_ols = model_ols.fit()

#    model_ramsaysE = RLM(endog, exog, M=norms.RamsayE())
#    results_ramsaysE = model_ramsaysE.fit(update_scale=False)

#    model_andrewWave = RLM(endog, exog, M=norms.AndrewWave())
#    results_andrewWave = model_andrewWave.fit(update_scale=False)

#    model_hampel = RLM(endog, exog, M=norms.Hampel(a=1.7,b=3.4,c=8.5)) # convergence problems with scale changed, not with 2,4,8 though?
#    results_hampel = model_hampel.fit(update_scale=False)

#######################
### Stack Loss Data ###
#######################
    from statsmodels.datasets.stackloss import load
    data = load()
    data.exog = sm.add_constant(data.exog)
#############
### Huber ###
#############
#    m1_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber1 = m1_Huber.fit()
#    m2_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber2 = m2_Huber.fit(cov="H2")
#    m3_Huber = RLM(data.endog, data.exog, M=norms.HuberT())
#    results_Huber3 = m3_Huber.fit(cov="H3")
##############
### Hampel ###
##############
#    m1_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel1 = m1_Hampel.fit()
#    m2_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel2 = m2_Hampel.fit(cov="H2")
#    m3_Hampel = RLM(data.endog, data.exog, M=norms.Hampel())
#    results_Hampel3 = m3_Hampel.fit(cov="H3")
################
### Bisquare ###
################
#    m1_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare1 = m1_Bisquare.fit()
#    m2_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare2 = m2_Bisquare.fit(cov="H2")
#    m3_Bisquare = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results_Bisquare3 = m3_Bisquare.fit(cov="H3")


##############################################
# Huber's Proposal 2 scaling                 #
##############################################

################
### Huber'sT ###
################
    m1_Huber_H = RLM(data.endog, data.exog, M=norms.HuberT())
    results_Huber1_H = m1_Huber_H.fit(scale_est=scale.HuberScale())
#    m2_Huber_H
#    m3_Huber_H
#    m4 = RLM(data.endog, data.exog, M=norms.HuberT())
#    results4 = m1.fit(scale_est="Huber")
#    m5 = RLM(data.endog, data.exog, M=norms.Hampel())
#    results5 = m2.fit(scale_est="Huber")
#    m6 = RLM(data.endog, data.exog, M=norms.TukeyBiweight())
#    results6 = m3.fit(scale_est="Huber")




#    print """Least squares fit
#%s
#Huber Params, t = 2.
#%s
#Ramsay's E Params
#%s
#Andrew's Wave Params
#%s
#Hampel's 17A Function
#%s
#""" % (results_ols.params, results_huber.params, results_ramsaysE.params,
#            results_andrewWave.params, results_hampel.params)

