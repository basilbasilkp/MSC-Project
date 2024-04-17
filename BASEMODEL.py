#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


matches = pd.read_csv('/Users/muhammedbasil/Downloads/matches.csv', index_col=0)


# In[4]:


matches.head()


# In[5]:


matches.shape


# In[6]:


matches["team"].value_counts()


# In[7]:


matches[matches["team"] == "Liverpool"]


# In[8]:


matches["round"].value_counts()


# In[9]:


matches.dtypes


# In[10]:


#coverting date into numerical value

matches["date"] = pd.to_datetime(matches["date"])


# In[11]:


matches.dtypes


# In[13]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[14]:


matches


# In[15]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes


# In[16]:


matches


# In[17]:


matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")


# In[ ]:





# In[19]:


matches["day_code"] = matches["date"].dt.dayofweek


# In[20]:


matches


# In[ ]:





# In[43]:


matches["result"]


# In[44]:


matches["target"] = (matches["result"] == 'W').astype("int")


# In[45]:


matches


# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[48]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split = 10, random_state = 1)


# In[49]:


train = matches[matches["date"] < '2022-01-01']


# In[50]:


test = matches[matches["date"] > '2022-01-01']


# In[51]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[52]:


rf.fit(train[predictors], train["target"])


# In[53]:


preds = rf.predict(test[predictors])


# In[54]:


#accuracy
from sklearn.metrics import accuracy_score


# In[55]:


acc = accuracy_score(test["target"], preds)


# In[56]:


acc


# In[57]:


#now lets see in which situations our accuracy was hoigh/low


# In[58]:


#we are combining our actual value and predicted value here 


# In[59]:


combined = pd.DataFrame(dict(actual = test["target"], prediction = preds))


# In[60]:


#create a table to understand


# In[61]:


pd.crosstab(index=combined["actual"], columns=combined["prediction"])


# In[62]:


#here predicting winn has less accuracy, so we are tring a new method


# In[63]:


from sklearn.metrics import precision_score


# In[65]:


precision_score(test["target"], preds)


# In[66]:


#our precision was only 47%


# In[67]:


#we need to improve this with ROLLING AVERAGES


# In[68]:


grouped_matches = matches.groupby("team")
#we are grouping all the data of a perticular team


# In[69]:


group = grouped_matches.get_group("Manchester City")


# In[70]:


#lets see the data of manchester city


# In[71]:


group


# In[72]:


#lets take match week 4. we are gonna look at how the team performed
#in match weeks 1, 2 and 3 and predict how they are gonna perform 
#in match week 4


# In[84]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)  
    return group


# In[85]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]


# In[86]:


new_cols


# In[88]:


rolling_averages(group, cols, new_cols)


# In[89]:


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))


# In[90]:


matches_rolling


# In[92]:


matches_rolling = matches_rolling.droplevel('team')


# In[93]:


matches_rolling


# In[94]:


#we have 1317 rows but only 36 index. so that means the index is
#repeating . we need unique value


# In[95]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[96]:


matches_rolling


# In[97]:


#so now we have some new predictors and 
#we need to use these predictors to improve the accuracy


# In[103]:


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index = test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


# In[104]:


combined, precision = make_predictions(matches_rolling, predictors + new_cols)


# In[105]:


precision


# In[106]:


combined


# In[107]:


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


# In[108]:


combined


# In[113]:


#here team names are written differently so we need to fix it
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
    
}
mapping = MissingDict(**map_values)
    


# In[114]:


mapping["West Ham United"]


# 

# In[115]:


combined["new_team"] = combined["team"].map(mapping)


# In[116]:


combined


# In[117]:


merged = combined.merge(combined, left_on = ["date", "new_team"], right_on=["date", "opponent"])


# In[118]:


merged


# In[121]:


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()


# In[122]:


27/40


# In[123]:


matches.columns


# In[ ]:




