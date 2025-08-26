import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

# 1. Download IBM stock data
ticker = 'IBM'
data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
data.to_csv('ibm_stock.csv')  # Save to CSV

# 2. Feature Engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=10).std()
data['Momentum'] = data['Close'] - data['Close'].shift(10)
data['Index'] = yf.download('^GSPC', start='2015-01-01', end='2024-01-01')['Close']
data['Index_Return'] = data['Index'].pct_change()
data['Index_Volatility'] = data['Index_Return'].rolling(window=10).std()
data['Index_Momentum'] = data['Index'] - data['Index'].shift(10)

# 3. Prepare dataset
data = data.dropna()
features = ['Volatility', 'Momentum', 'Index_Volatility', 'Index_Momentum']
X = data[features]
y = np.where(data['Return'].shift(-1) > 0, 1, 0)  # 1: Up, 0: Down

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. SVM Model (RBF Kernel)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# 7. Prediction
y_pred = svm.predict(X_test_scaled)

# 8. Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Plot results
plt.figure(figsize=(12,6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Trend')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Trend', alpha=0.7)
plt.legend()
plt.title('IBM Stock Trend Prediction (SVM-RBF)')
plt.xlabel('Date')
plt.ylabel('Trend (1=Up, 0=Down)')
plt.show()