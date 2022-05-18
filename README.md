# Efficient-Frontier
We implement an Efficient Frontier that plots the optimal expected portfolio return against a certain level of risk measured by portfolio variance. This is achieved by 
constraint optimisation using predefined portfolio constraints and iterating across expected levels of risk from the Lowest-Variance portfolio to the variance of the 
portfolio with the maximum Sharpe Ratio. We define various constraints to calculate the weightings for the minVar portfolio(Minimum Variance), the MaxSR(Maximum Sharpe
Ratio Portfolio) and the optimisation required to calculate the optimal return weightings for all the variance values that lie between. This requires us to define portfolio performance functions which will be the focus of our code in the beginning. 

Firstly we'd need to set up the covariance matrix and the expected returns of the multiple stocks to be added to multi-stock porttfolio. We extract data using the pandas datareader(Yahoo) and set up the requisites in the following section.

```python
#Importing Data from YFin
def getData(stocks, start, end):
  stockData = pdr.get_data_yahoo(stocks, start = start, end = end)
  stockData = stockData['Close']

  returns = stockData.pct_change()
  meanReturns = returns.mean()
  covMatrix = returns.cov()
  return meanReturns, covMatrix
```
![image](https://user-images.githubusercontent.com/51220035/169164116-2f15a885-cdfd-4692-aded-f97400d535b1.png)
![Uploading image.png…]()
