#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Mission 1:
# 
# * Calculate the Average Rating according to current comments and compare it with the existing average rating.

# In[2]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/amazon_review.csv"

amazon_review = pd.read_csv(path)


# In[75]:


df = amazon_review.copy()
print("DataFrame Shape : {}".format(df.shape))


# In[76]:


df.head()


# In[77]:


# Are there any missing values?
# Can be ignored !!!

na_values = df.isnull().sum()
na_values = pd.DataFrame(na_values)
na_values.columns = ["Na_Values"]
na_values = na_values[na_values["Na_Values"] > 0 ]
na_values


# In[78]:


dtypes = pd.DataFrame(df.dtypes).reset_index()
dtypes.columns = ["Name", "Dtype"]
dtypes


# In[79]:


df.head()


# In[80]:


# We changed our Review variable to datetime....

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.dtypes


# In[81]:


# unixReviewTime variables review Time has the same meaning... 
# With pd.to_datetime(df["unix Review Time"] , unit = "s") we can see that ....

time_df = pd.DataFrame()
time_df["reviewTime"] = pd.to_datetime(df["reviewTime"])
time_df["unixReviewTime"] = pd.to_datetime(df["unixReviewTime"], unit="s")
time_df.dtypes


# In[82]:


time_df.head()


# In[83]:


nunique = pd.DataFrame(df.nunique()).reset_index()
nunique.columns = ["Names" , "Nunique"]
nunique.sort_values("Nunique", ascending = False)


# In[84]:


# So I'll cover the required columns...
# Daydiff calculated for us but we recalculate ... (1 more than max day ...)
# [1,2] : Here the first one represents helpful_yes while the other one represents total_vote..
# So df["total_vote"] - df['helpful_yes'] = df["helpful_no"]

df[df["total_vote"] > 0][["helpful","helpful_yes","total_vote"]].head()


# In[85]:


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df = df[["reviewText", "reviewTime", "overall", "summary", "helpful_yes","helpful_no","total_vote"]]
df.head()


# In[86]:


print("Max Overall : {}".format(df["overall"].max()) )
print("Min Overall : {}".format(df["overall"].min()) )


# In[87]:


# Is the average really for this product?

print("Average Overall : {}".format(df["overall"].mean()))


# In[88]:


print("Max Time : {}".format(df["reviewTime"].max()))
print("Min Time : {}".format(df["reviewTime"].min()))


# In[89]:


import datetime as dt
current_date = pd.to_datetime("2014-12-8")
current_date


# In[97]:


df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()


# In[99]:


df.sort_values("reviewTime", ascending = False).head()


# In[102]:


# Let's format it by time...
# We have weighted by time intervals ...

def time_based_weighted_average(dataframe , w1 = 28 , w2 = 26, w3 = 24, w4 = 22):
    
    results =     df.loc[ df["days"] <= 60, "overall" ].mean() * w1 / 100 +     df.loc[ (df["days"] > 60) & (df["days"] <=120), "overall"].mean() * w2 / 100 +     df.loc[ (df["days"] > 120) & (df["days"] <=180) , "overall"].mean() * w3 / 100 +     df.loc[ (df["days"] > 180),"overall"].mean() * w4 / 100
    
    return results


# In[103]:


print("General Overall : {}".format(df.overall.mean())) 
print("By Time : {}".format(time_based_weighted_average(df)))


# ## Mission 2:
# * Specify 20 reviews to be displayed on the product detail page for the product.

# In[105]:


################################################ #
# Wilson Lower Bound Score
################################################ #


# Helpful / Not Found
# Liked / Disliked
# Thumbs Up/ Thumbs Down
# up /down

# p : up rate

# For example, if there is 600 up to 400 down interaction, what is the up rate?
# 600/1000 is our 0.6 up rate. so p is our hat value.


# It will be at this rate with 0.05 deviation between 0.5 and 0.7 instead of 0.6 !!!

def wilson_lower_bound(up, down, confidence=0.95):
    
    """
    Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be adjusted to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.
    
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    import scipy.stats as st
    import math
    
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# In[106]:


def score_up_down_diff(up,down):
    return up - down


# In[107]:


def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up + down)


# In[108]:


#If Helpful hadn't been given to us in pieces we could have separated it by doing so....

# df.helpful = df.helpful.apply(lambda x : x.strip("[]").split(","))
# df["helpful_yes"] = df.helpful.apply(lambda x : x[0]).astype("int64")
# df["total_reviews"] = df.helpful.apply(lambda x : x[1]).astype("int64")
# df["helpful_no"] = df["total_reviews"] - df["helpful_yes"]


# In[109]:


# Score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x : score_up_down_diff(x["helpful_yes"],x["helpful_no"]), axis = 1)

# Score_average_rating
df["score_average_rating"] = df.apply(lambda x : score_average_rating(x["helpful_yes"],x["helpful_no"]), axis = 1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x : wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis = 1)


# In[110]:


# Ranking by score_pos_neg_diff and score_average_rating can give us wrong information.
# Or it may cause the ones that should stand out not to come out...
# Thus, we can be hesitant to buy or get frustrated because of the information we can't get about the product...

df.sort_values("wilson_lower_bound" , ascending = False).head(20)


# In[ ]:




