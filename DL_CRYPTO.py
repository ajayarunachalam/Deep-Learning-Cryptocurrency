# In[1]:

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[2]:

# get market info for bitcoin from the start of 2017 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170101&end="+time.strftime("%Y%m%d"))[0]


# In[3]:

bitcoin_market_info.shape


# In[4]:

bitcoin_market_info.columns


# In[5]:

bitcoin_market_info.head(2)


# In[6]:

# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))


# In[7]:

bitcoin_market_info.head(2)


# In[15]:

# get market info for zcoin from the start of 2017 to the current day
zcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/zcoin/historical-data/?start=20170101&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
zcoin_market_info = zcoin_market_info.assign(Date=pd.to_datetime(zcoin_market_info['Date']))
# look at the first few rows
zcoin_market_info.head(1)


# In[18]:

zcoin_market_info.dtypes


# In[19]:

# function for Converting the prices from usd to thai baht - current price as of 8.3.2018 (1 USD = 31.30)

def usd2baht(x):
    # that, if x is a string,
    if type(x) is float:
        #return it multiplied by 31.30
        return 31.30 * x
    # but, if not, 
    elif x:
        # just returns it untouched
        return x
    # and leave everything else
    else:
        return


# In[22]:

zcoin_market_info_baht = zcoin_market_info.applymap(usd2baht)


# In[23]:

zcoin_market_info_baht.head(1)

# Summary of steps 1- 23
# We've loaded some python packages and then imported the data from the coinmarketcap site. 
# With a little bit of data cleaning, we arrive at the above table. 
# We also do the same thing for zcoin by simply replacing 'bitcoin' with 'zcoin' in the initial url.

# To prove that the data is accurate, we can plot the price and volume of both cryptos over time.

# In[28]:

# getting the Bitcoin and ZCOIN logos
import sys
from PIL import Image
import io

if sys.version_info[0] < 3:
    import urllib2 as urllib
    bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
   
else:
    import urllib
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
   

image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

# In[29]:

zcoin_market_info_baht.columns


# In[30]:

bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
zcoin_market_info_baht.columns =[zcoin_market_info_baht.columns[0]]+['xzc_'+i for i in zcoin_market_info_baht.columns[1:]]


# In[42]:

fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
ax1.set_ylabel('Closing Price (Thai Baht)',fontsize=12)
ax2.set_ylabel('Volume (Thai Baht)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2017,2019) for j in [1,7]])
ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),bitcoin_market_info['bt_Open'])
ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
fig.tight_layout()
fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)
plt.show()


# In[35]:

fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
ax1.set_ylabel('Closing Price (THAI BAHT)',fontsize=12)
ax2.set_ylabel('Volume (THAI BAHT)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2017,2019) for j in [1,7]])
ax1.plot(zcoin_market_info_baht['Date'].astype(datetime.datetime),zcoin_market_info_baht['xzc_Open'])
ax2.bar(zcoin_market_info_baht['Date'].astype(datetime.datetime).values, zcoin_market_info_baht['xzc_Volume'].values)
fig.tight_layout()
plt.show()


# In[36]:

# Converting bitcoin price in USD to baht
bitcoin_market_info_baht = bitcoin_market_info.applymap(usd2baht)


# In[37]:

bitcoin_market_info_baht.head(1)


# In[38]:

market_info = pd.merge(bitcoin_market_info_baht,zcoin_market_info_baht, on=['Date'])
market_info = market_info[market_info['Date']>='2017-01-01']
for coins in ['bt_', 'xzc_']: 
    kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head(1)


# # Split into training and test sets. 
# The model is built on the training set and subsequently evaluated on the unseen test set. 
# Rather arbitrarily, I have set the cut-off date to Dec 1st 2017 (i.e. model will be trained on data before that date and validated on data after it).

# In[43]:

split_date = '2017-12-01'
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2017,2019) for j in [1,7]])
ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['bt_Close'], 
         color='#B08FC7', label='Training')
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['bt_Close'], 
         color='#8FBAC8', label='Test')
ax2.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['xzc_Close'], 
         color='#B08FC7')
ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['xzc_Close'], color='#8FBAC8')
ax1.set_xticklabels('')
ax1.set_ylabel('Bitcoin Price (Thai Baht)',fontsize=12)
ax2.set_ylabel('ZCOIN Price (Thai Baht)',fontsize=12)
plt.tight_layout()
ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
fig.figimage(bitcoin_im.resize((int(bitcoin_im.size[0]*0.65), int(bitcoin_im.size[1]*0.65)), Image.ANTIALIAS), 
             200, 260, zorder=3,alpha=.5)
plt.show()


# Before we take our deep machine learning model, it's worth discussing a simpler model. 
# The most basic model is to set tomorrow's price equal to today's price (which we'll crudely call a lag model). 

# In[47]:

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['bt_Close'][1:].values, label='Predicted')
ax1.set_ylabel('Bitcoin Price (Thai Baht)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.set_title('Simple Lag Model (Test Set)')
ax2.set_ylabel('ZCOIN Price (Thai Baht)',fontsize=12)
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['xzc_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['xzc_Close'][1:].values, label='Predicted')
fig.tight_layout()
plt.show()


# First, we may want to make sure the daily change in price follows a normal distribution. We'll plot the histogram of values.
# In[48]:

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(market_info[market_info['Date']< split_date]['bt_day_diff'].values, bins=100)
ax2.hist(market_info[market_info['Date']< split_date]['xzc_day_diff'].values, bins=100)
ax1.set_title('Bitcoin Daily Price Changes')
ax2.set_title('ZCOIN Daily Price Changes')
plt.show()


# # SINGLE POINT RANDOM WALK TEST

# In[52]:

np.random.seed(202)
bt_r_walk_mean, bt_r_walk_sd = np.mean(market_info[market_info['Date']< split_date]['bt_day_diff'].values), np.std(market_info[market_info['Date']< split_date]['bt_day_diff'].values)
bt_random_steps = np.random.normal(bt_r_walk_mean, bt_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)
xzc_r_walk_mean, xzc_r_walk_sd = np.mean(market_info[market_info['Date']< split_date]['xzc_day_diff'].values), np.std(market_info[market_info['Date']< split_date]['xzc_day_diff'].values)
xzc_random_steps = np.random.normal(xzc_r_walk_mean, xzc_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
     market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['bt_Close'].values[1:] * 
     (1+bt_random_steps), label='Predicted')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
     market_info[market_info['Date']>= split_date]['xzc_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['xzc_Close'].values[1:] * 
     (1+xzc_random_steps), label='Predicted')
ax1.set_title('Single Point Random Walk (Test Set)')
ax1.set_ylabel('Bitcoin Price (THAI BAHT)',fontsize=12)
ax2.set_ylabel('ZCOIN Price (THAI BAHT)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()


# Apart from a few differences, it broadly tracks the actual closing price for each coin. 
# It even captures the ZCOIN rises (and subsequent falls) in mid-June and late August. 
# Single point predictions are unfortunately quite common when evaluating time series models. 
# A better idea could be to measure its accuracy on multi-point predictions. That way, errors from previous predictions aren't reset but rather are compounded by subsequent predictions. Thus, poor models are penalised more heavily. In mathematical terms:
# 
# Let's get our random walk model to predict the closing prices over the total test set.

# In[54]:

bt_random_walk = []
xzc_random_walk = []
for n_step, (bt_step, xzc_step) in enumerate(zip(bt_random_steps, xzc_random_steps)):
    if n_step==0:
        bt_random_walk.append(market_info[market_info['Date']< split_date]['bt_Close'].values[0] * (bt_step+1))
        xzc_random_walk.append(market_info[market_info['Date']< split_date]['xzc_Close'].values[0] * (xzc_step+1))
    else:
        bt_random_walk.append(bt_random_walk[n_step-1] * (bt_step+1))
        xzc_random_walk.append(xzc_random_walk[n_step-1] * (xzc_step+1))
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         bt_random_walk[::-1], label='Predicted')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['xzc_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         xzc_random_walk[::-1], label='Predicted')

ax1.set_title('Full Interval Random Walk')
ax1.set_ylabel('Bitcoin Price (THAI BAHT)',fontsize=12)
ax2.set_ylabel('ZCOIN Price (THAI BAHT)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()


# The model predictions are extremely sensitive to the random seed. I've selected one where the full interval random walk looks almost decent for ZCOIN.

# In[56]:

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def plot_func(freq):
    np.random.seed(freq)
    random_steps = np.random.normal(xzc_r_walk_mean, xzc_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)
    random_walk = []
    for n_step,i in enumerate(random_steps):
        if n_step==0:
            random_walk.append(market_info[market_info['Date']< split_date]['xzc_Close'].values[0] * (i+1))
        else:
            random_walk.append(random_walk[n_step-1] * (i+1))
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['xzc_Close'].values, label='Actual')
    ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['xzc_Close'].values[1:] * 
         (1+random_steps), label='Predicted')
    ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['xzc_Close'].values[1:] * 
         (1+random_steps))
    ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
             random_walk[::-1])
    ax1.set_title('Single Point Random Walk')
    ax1.set_ylabel('')
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    ax2.set_title('Full Interval Random Walk')
    fig.text(0.0, 0.5, 'ZCOIN Price (THAI BAHT)', va='center', rotation='vertical',fontsize=12)
    plt.tight_layout()
    plt.show()
    
interact(plot_func, freq =widgets.IntSlider(min=50,max=100,step=1,value=50, description='Random Seed:'))


# Notice how the single point random walk always looks quite accurate, even though there's no real substance behind it. 
# Hopefully, you'll be more suspicious of any blog that claims to accurately predict prices. 

# # Long Short Term Memory (LSTM)
# To build the LSTM network, 
# there exists packages that include standard implementations of various deep learning algorithms (e.g. TensorFlow, Keras, PyTorch, etc.). 
# I'll opt for Keras, as it is simple for non-experts. 

# In[57]:

for coins in ['bt_', 'xzc_']: 
    kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
    market_info = market_info.assign(**kwargs)


# In[58]:

market_info.shape


# In[59]:

market_info.columns


# In[60]:

market_info.head(2)


# In[61]:

model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'xzc_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')
model_data.head(2)


# I've created a new data frame called model_data. 
# I've removed some of the previous columns (open price, daily highs and lows) and reformulated some new ones. 
# close_off_high represents the gap between the closing price and price high for that day, where values of -1 and 1 mean the closing price was equal 
# to the daily low or daily high, respectively. 

# The volatility columns are simply the difference between high and low price divided by the opening price. 
# You may also notice that model_data is arranged in order of earliest to latest. 
 
# We don't actually need the date column anymore, as that information won't be fed into the model.
# 
# Our LSTM model will use previous data (both bitcoin and xzc) to predict the next day's closing price of a specific coin. 
# We must decide how many previous days it will have access to. Again, it's rather arbitrary, but I'll opt for 10 days, as it's a nice round number. 
# We build little data frames consisting of 10 consecutive days of data (called windows), so the first window will consist of the 0-9th rows of the training set 

# Deep learning models don't like inputs that vary wildly. 
# Looking at those columns, some values range between -1 and 1, while others are on the scale of millions. 

# We need to normalise the data, so that our inputs are somewhat consistent. Typically, you want values between -1 and 1. 
# The off_high and volatility columns are fine as they are. For the remaining columns, we'll normalise the inputs to the first value in the window.

# In[62]:

# we don't need the date columns anymore
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)


# In[63]:

window_len = 10
norm_cols = [coin+metric for coin in ['bt_', 'xzc_'] for metric in ['Close','Volume']]


# In[64]:

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['xzc_Close'][window_len:].values/training_set['xzc_Close'][:-window_len].values)-1


# In[65]:

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['xzc_Close'][window_len:].values/test_set['xzc_Close'][:-window_len].values)-1


# In[66]:

LSTM_training_inputs[0]


# In[67]:

LSTM_test_inputs[0]

# We're now ready to build the LSTM model. 

# In[68]:

# Its easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


# In[71]:

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[72]:

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# So, the build_model functions constructs an empty model unimaginatively called model (model = Sequential), 
# to which an LSTM layer is added. 
# That layer has been shaped to fit our inputs (n x m tables, where n and m represent the number of timepoints/rows and columns, respectively). 

# The function also includes more generic neural network features, like dropout and activation functions. 
# Now, we just need to specify the number of neurons to place in the LSTM layer (I've opted for 20 to keep runtime reasonable), 
# as well as the data on which the model will be trained.

# In[73]:

# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
xzc_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['xzc_Close'][window_len:].values/training_set['xzc_Close'][:-window_len].values)-1
# train model on data
# note: xzc_history contains information on the training error per epoch
xzc_history = xzc_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


# If everything went to plan, then we'd expect the training error to have gradually decreased over time.

# In[74]:

fig, ax1 = plt.subplots(1,1)

ax1.plot(xzc_history.epoch, xzc_history.history['loss'])
ax1.set_title('Training Error')

if xzc_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
# just in case you decided to change the model loss calculation
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()


# We've just built an LSTM model to predict tomorrow's ZCOIN closing price. 
# Let's see how well it performs. We start by examining its performance on the training set (data before DEC 2017). 
# That number below the code represents the model's mean absolute error (mae) on the training set after the 50th training iteration (or epoch). 
# Instead of relative changes, we can view the model output as daily closing prices.

# In[78]:

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,5,9]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2017,2019) for j in [1,5,9]])
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['xzc_Close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(xzc_model.predict(LSTM_training_inputs))+1) * training_set['xzc_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('ZCOIN Price (THAI BAHT)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(xzc_model.predict(LSTM_training_inputs))+1)-            (training_set['xzc_Close'].values[window_len:])/(training_set['xzc_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
axins.set_xticks([datetime.date(i,j,1) for i in range(2017,2019) for j in [1,5,9]])
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['xzc_Close'][window_len:], label='Actual')
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(xzc_model.predict(LSTM_training_inputs))+1) * training_set['xzc_Close'].values[:-window_len])[0], 
         label='Predicted')
axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
axins.set_ylim([10,60])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()


# We shouldn't be too surprised by its apparent accuracy here. 
# The model could access the source of its error and adjust itself accordingly. 
# In fact, it's not hard to attain almost zero training errors. We could just cram in hundreds of neurons and train for thousands of epochs 
# (a process known as overfitting, where you're essentially predicting noise- I included the Dropout() call in the build_model function to mitigate this risk for our relatively small model). 
# We should be more interested in its performance on the test dataset, as this represents completely new data for the model.

# In[80]:

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['xzc_Close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(xzc_model.predict(LSTM_test_inputs))+1) * test_set['xzc_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(xzc_model.predict(LSTM_test_inputs))+1)-(test_set['xzc_Close'].values[window_len:])/(test_set['xzc_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('ZCOIN Price (THAI BAHT)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()


# Our LSTM model seems to have performed well on the unseen test set. 

# In[93]:

xzc_model.save('xzc_model_randseed_%d.h5')


# # LSTM Model for bitcoin

# In[81]:

# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
bt_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# train model on data
# note: nt_history contains information on the training error per epoch
bt_history = bt_model.fit(LSTM_training_inputs, 
                            (training_set['bt_Close'][window_len:].values/training_set['bt_Close'][:-window_len].values)-1, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


# In[82]:

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['bt_Close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_training_inputs))+1) * training_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('Bitcoin Price (Thai Baht)',fontsize=12)
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(bt_model.predict(LSTM_training_inputs))+1)- (training_set['bt_Close'].values[window_len:])/(training_set['bt_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})

axins = zoomed_inset_axes(ax1, 2.52, loc=10, bbox_to_anchor=(400, 307)) 
axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['bt_Close'][window_len:], label='Actual')
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_training_inputs))+1) * training_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')
axins.set_xlim([datetime.date(2017, 2, 15), datetime.date(2017, 5, 1)])
axins.set_ylim([920, 1400])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()


# In[84]:

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][10:].astype(datetime.datetime),
         test_set['bt_Close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][10:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_test_inputs))+1) * test_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(bt_model.predict(LSTM_test_inputs))+1)- (test_set['bt_Close'].values[window_len:])/(test_set['bt_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('Bitcoin Price (Thai Baht)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()


# As I've stated earlier, single point predictions can be deceptive. 
# Looking more closely, you'll notice that, again, the predicted values regularly mirror the previous values (e.g. October). 
# Our fancy deep learning LSTM model has partially reproducted a autregressive (AR) model of some order p, where future values are simply the weighted sum of the previous p values.
# The good news is that AR models are commonly employed in time series tasks (e.g. stock market prices), so the LSTM model appears to have landed on a sensible solution. The bad news is that it's a waste of the LSTM capabilities, we could have a built a much simpler AR model in much less time and probably achieved similar results (though the title of this post would have been much less clickbaity). More complex does not automatically equal more accurate).

# In[94]:

bt_model.save('bt_model_randseed_%d.h5')


# # We'll now build LSTM models to predict crypto prices for the next 5 days.

# In[85]:

# random seed for reproducibility
np.random.seed(202)
# we'll try to predict the closing price for the next 5 days 
# change this value if you want to make longer/shorter prediction
pred_range = 5
# initialise model architecture
xzc_model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
# model output is next 5 prices normalised to 10th previous closing price
LSTM_training_outputs = []
for i in range(window_len, len(training_set['xzc_Close'])-pred_range):
    LSTM_training_outputs.append((training_set['xzc_Close'][i:i+pred_range].values/
                                  training_set['xzc_Close'].values[i-window_len])-1)
LSTM_training_outputs = np.array(LSTM_training_outputs)
# train model on data
# note: xzc_history contains information on the training error per epoch
xzc_history = xzc_model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


# In[86]:

# random seed for reproducibility
np.random.seed(202)
# we'll try to predict the closing price for the next 5 days 
# change this value if you want to make longer/shorter prediction
pred_range = 5
# initialise model architecture
bt_model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
# model output is next 5 prices normalised to 10th previous closing price
LSTM_training_outputs = []
for i in range(window_len, len(training_set['bt_Close'])-pred_range):
    LSTM_training_outputs.append((training_set['bt_Close'][i:i+pred_range].values/
                                  training_set['bt_Close'].values[i-window_len])-1)
LSTM_training_outputs = np.array(LSTM_training_outputs)
# train model on data
# note: bt_history contains information on the training error per epoch
bt_history = bt_model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


# In[87]:

# little bit of reformatting the predictions to closing prices
xzc_pred_prices = ((xzc_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)* test_set['xzc_Close'].values[:-(window_len + pred_range)][::5].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))
bt_pred_prices = ((bt_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)* test_set['bt_Close'].values[:-(window_len + pred_range)][::5].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))

pred_colors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
ax2.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['bt_Close'][window_len:], label='Actual')
ax2.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['xzc_Close'][window_len:], label='Actual')
for i, (xzc_pred, bt_pred) in enumerate(zip(xzc_pred_prices, bt_pred_prices)):
    # Only adding lines to the legend once
    if i<5:
        ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                 bt_pred, color=pred_colors[i%5], label="Predicted")
    else: 
        ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                 bt_pred, color=pred_colors[i%5])
    ax2.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
             xzc_pred, color=pred_colors[i%5])
ax1.set_title('Test Set: 5 Timepoint Predictions',fontsize=13)
ax1.set_ylabel('Bitcoin Price (THAI BAHT)',fontsize=12)
ax1.set_xticklabels('')
ax2.set_ylabel('ZCOIN Price (THAI BAHT)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.13, 1), loc=2, borderaxespad=0., prop={'size': 12})
fig.tight_layout()
plt.show()

# In[95]:

# We calculate the average mean absolute error (mae)

from keras.models import load_model

xzc_preds = []
bt_preds = []
for rand_seed in range(775,800):
    temp_model = load_model('xzc_model_randseed_%d.h5'%rand_seed)
    xzc_preds.append(np.mean(abs(np.transpose(temp_model.predict(LSTM_test_inputs))-
                (test_set['xzc_Close'].values[window_len:]/test_set['xzc_Close'].values[:-window_len]-1))))
    temp_model = load_model('bt_model_randseed_%d.h5'%rand_seed)
    bt_preds.append(np.mean(abs(np.transpose(temp_model.predict(LSTM_test_inputs))-
                (test_set['bt_Close'].values[window_len:]/test_set['bt_Close'].values[:-window_len]-1))))


# In[98]:

xzc_preds = []
bt_preds = []
xzc_preds.append(np.mean(abs(np.transpose(xzc_model.predict(LSTM_test_inputs))-
                (test_set['xzc_Close'].values[window_len:]/test_set['xzc_Close'].values[:-window_len]-1))))
   
bt_preds.append(np.mean(abs(np.transpose(bt_model.predict(LSTM_test_inputs))-
                (test_set['bt_Close'].values[window_len:]/test_set['bt_Close'].values[:-window_len]-1))))


# In[99]:

print(xzc_preds)


# In[100]:

print(bt_preds)


# In[90]:

xzc_random_walk_preds = []
bt_random_walk_preds = []
for rand_seed in range(775,800):
    np.random.seed(rand_seed)
    xzc_random_walk_preds.append(
        np.mean(np.abs((np.random.normal(xzc_r_walk_mean, xzc_r_walk_sd, len(test_set)-window_len)+1)-
                       np.array(test_set['xzc_Close'][window_len:])/np.array(test_set['xzc_Close'][:-window_len]))))
    bt_random_walk_preds.append(
    np.mean(np.abs((np.random.normal(bt_r_walk_mean, bt_r_walk_sd, len(test_set)-window_len)+1)-
                       np.array(test_set['bt_Close'][window_len:])/np.array(test_set['bt_Close'][:-window_len]))))


# In[92]:

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.boxplot([bt_preds, bt_random_walk_preds],widths=0.75)
ax1.set_ylim([0, 0.2])
ax2.boxplot([xzc_preds, xzc_random_walk_preds],widths=0.75)
ax2.set_ylim([0, 0.2])
ax1.set_xticklabels(['LSTM', 'Random Walk'])
ax2.set_xticklabels(['LSTM', 'Random Walk'])
ax1.set_title('Bitcoin Test Set (25 runs)')
ax2.set_title('ZCOIN Test Set (25 runs)')
ax2.set_yticklabels('')
ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
plt.show()


# The graphs show the error on the test set after 25 different initialisations of each model. 
# The LSTM model returns an average error of about 0.04 and 0.05 on the bitcoin and xzc prices, respectively, better than the corresponding random walk models.

# # Summary
# We've collected some crypto data and build a deep learning model. 
# If past prices alone are sufficient to decently forecast future prices, we need to include other features that provide comparable predictive model. 
# That way, the LSTM model wouldn't be so reliant on past prices, potentially unlocking more complex behaviours. 
