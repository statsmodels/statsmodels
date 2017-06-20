from summary_stats import SurveyStat

# 15 PSUs, number of stratum = 1, two variables, includes prob_weights
df = pd.read_csv("examples/survey_df.csv")

survey_df = SurveyStat(df, cluster="dnum", prob_weights="pw")

survey_tot = survey_df.total('jack') # matches perfectly with R result

# returns tuple of two arrays. First array is the estimates, second is the corresponding SEs
# R returns
#         total     SE
# api99 3759623 856867
# api00 3989986 907399

# we return
#        total            SE
# api99 3759622.80883407 856866.80598719
# api00 3989985.46570205 907398.70559744
print(survey_tot) 


survey_mean = survey_df.mean('jack') 
# R returns
#         mean     SE
# api99 606.98 24.469
# api00 644.17 23.779

# This returns
#         mean     SE
# api99 606.98 27.27487881
# api00 644.17 26.59966787
print(survey_mean) 

print(survey_df.percentile(25)) 
# R returns 
#         0.25 
# api00 551.75 
# api99 512.00 

# this returns 
#         0.25 
# api00 552.
# api99 512.

print(survey_df.median()) 
# R returns 
#         0.5 
# api00 652 
# api99 615

# this returns
#         0.5 
# api00 652.
# api99 615.