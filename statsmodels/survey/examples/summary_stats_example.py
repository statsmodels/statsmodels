import summary_stats
# 15 PSUs, number of stratum = 1, two variables, includes prob_weights
df = pd.read_csv("examples/survey_df.csv")

survey_df = SurveyDesign(np.ones(df.shape[0]), np.asarray(df["dnum"]), np.asarray(df["pw"]))

survey_tot = SurveyTotal(survey_df, df, 'jack') # matches perfectly with R result
print(survey_tot.est)
print(survey_tot.vc)
# returns tuple of two arrays. First array is the estimates, second is the corresponding SEs
# R returns
#         total     SE
# api99 3759623 856867
# api00 3989986 907399

# we return
#        total            SE
# api99 3759622.80883407 856866.80598719
# api00 3989985.46570205 907398.70559744

survey_mean = SurveyMean(survey_df, df, 'jack')
print(survey_mean.est)
print(survey_mean.vc) # does not match up w/ R but it does match up in the dummy example

strata = np.r_[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
cluster = np.r_[0, 0, 2, 2, 3, 3, 4, 4, 4, 6, 6, 6]
weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2].astype(np.float64)
data = np.asarray([[1, 3, 2, 5, 4, 1, 2, 3, 4, 6, 9, 5],
                   [5, 3, 2, 1, 4, 7, 8, 9, 5, 4, 3, 5]], dtype=np.float64).T

design = SurveyDesign(strata, cluster, weights)
avg = SurveyMean(design,data, 'jack')
tot = SurveyTotal(design, data, 'jack')
assert(np.allclose(avg.est, np.r_[3.777778, 4.722222]))
assert(np.allclose(avg.vc, np.r_[0.9029327, 1.061515]))
import os
print(os.listdir())


# R returns
#         mean     SE
# api99 606.98 24.469
# api00 644.17 23.779

# This returns
#         mean     SE
# api99 606.98 27.27487881
# api00 644.17 26.59966787

# need to fix this. get some keyerror
perc = SurveyQuantile(survey_df, df, [.25, .50,1,2])
print(perc.est)
# R returns 
#         0.25 
# api00 551.75 
# api99 512.00 

# this returns 
#         0.25 
# api00 552.
# api99 512.

med = SurveyMedian(survey_df, df)
print(med.est)
# R returns 
#         0.5 
# api00 652 
# api99 615

# this returns
#         0.5 
# api00 652.
# api99 615.