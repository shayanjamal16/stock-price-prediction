#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.layers import Dropout


# In[16]:


data = pd.read_csv('nifty50_latest_data.csv')


# In[17]:


data


# In[35]:


X = data.iloc[:,[1,2,3,4]].values.astype(float)
y = data.iloc[:,[4]].values.astype(float)


# In[36]:


y


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[38]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[39]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)


# In[40]:


def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


# In[44]:


#ANN Model of Selection by Syed Laraib
model = Sequential()
model.add(Dense(units=1500, activation='relu', input_dim=4))  # Input layer
model.add(Dropout(0.2))
model.add(Dense(units=1250, activation='relu'))  # Hidden Layer 1
model.add(Dropout(0.2))
model.add(Dense(units=1000, activation='relu'))  # Hidden layer 2
model.add(Dropout(0.2))
model.add(Dense(units=750, activation='relu'))  # Hidden layer 3
model.add(Dropout(0.2))
model.add(Dense(units=500, activation='relu'))  # Hidden layer 4
model.add(Dropout(0.2))
model.add(Dense(units=250, activation='relu'))  # Hidden layer 5
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear')) #Output layer


# In[47]:


model.compile(loss='mean_squared_logarithmic_error', optimizer=Nadam(learning_rate=0.0001), metrics=[r_squared])


# In[48]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=16, verbose=1, callbacks=[reduce_lr])


# In[49]:


model.save_weights('model_stocks.h5')
model.load_weights('model_stocks.h5')


# In[29]:


y_pred = model.predict(X_test)
print(y_pred)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.to_csv('predictions.csv', index=False)


# In[50]:


plt.figure(figsize=(12, 4))


# In[51]:


# Predicted vs Actual graph
plt.subplot(1, 1, 1)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(y_pred, color='red', label='Predicted')
plt.title('Predicted vs Actual')
plt.legend()



# In[55]:


plt.figure(figsize=(15, 10))

# Assuming y_test and y_pred are numpy arrays
for i in range(1):
    plt.subplot(3, 1, i+1)

    # Plot histogram for Actual values of output i
    plt.hist(y_test[:, i], bins=30, alpha=0.5, color='blue', label='Actual')

    # Plot histogram for Predicted values of output i
    plt.hist(y_pred[:, i], bins=30, alpha=0.5, color='red', label='Predicted')

    plt.title(f'Actual vs Predicted Histogram for Output {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()


# In[56]:


# Accuracy graph
plt.subplot(2, 1, 1)
plt.plot(history.history['r_squared'], label='train')
plt.plot(history.history['val_r_squared'], label='test')
plt.title('Accuracy')
plt.legend()


# In[57]:


# Loss graph
plt.subplot(3, 1, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




