import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV


# Pretty display for notebooks
####%matplotlib inline

# Load the Boston housing dataset
boston_data = datasets.load_boston()
prices = boston_data.target
features = boston_data.data    
print "Boston data load successfully:"

### statistical analysis and data exploration
minimum_price = np.amin(prices)
maximum_price = np.amax(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_dev = np.std(prices, ddof=1)
print "Housing data stats:"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_dev)

CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
def shuffle_split_data(x, y):
   ####split the data
   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
   
   return x_train,y_train,x_test,y_test

try:
   x_train, y_train, x_test, y_test = shuffle_split_data(features, prices)
   print "successfully shuffled and split the data!"
except:
   print "error while splitting the data"

def performance_metric(true, predict):
   total_error = metrics.mean_squared_error(true, predict)
   return total_error
try:
   total_error = performance_metric(y_train, y_predict)
   print "successfully performed a metric calculation:"
except:
   print "error while calculating performance metric"

def fit_model(x, y):
   regressor = DecisionTreeRegressor(random_state=42)
   parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
   scoring_function = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better = False)
   reg  = GridSearchCV(regressor, parameters, scoring = scoring_function)
   reg.fit(x, y)
   return reg

try:
   reg = fit_model(features, prices)
   print "Successfully fit a model!"
except:
   print "something went wrong"

def learning_curves(X_train, y_train, X_test, y_test):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . ."
    
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()



def model_complexity(X_train, y_train, X_test, y_test):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    print "Creating a model complexity graph. . . "

    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()
## ce=reating learning curves
learning_curves(x_train, y_train, x_test, y_test)

##create a model complexity graph
model_complexity(x_train, y_train, x_test, y_test)

print "Final model has an optimal max_depth parameter of", reg.best_params_

sale_price = reg.predict(CLIENT_FEATURES)
print "Predicted value of client's home: {0:.3f}".format(sale_price[0])