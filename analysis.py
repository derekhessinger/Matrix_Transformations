'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Derek Hessinger
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        minArr = [] # create an empty list to hold the data

        vData = self.data.select_data(headers, rows)    # use select_data to grab variable data
        minArr = np.amin(vData, axis=0)     # use amin to find the minimum value in each column
        minArr = np.array(minArr)       # turn the list into an np array

        return minArr

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        maxArr = [] # create an empty list to hold the data

        vData = self.data.select_data(headers, rows)    # use select_data to grab variable data
        maxArr = np.amax(vData, axis=0)     # use amax to find the maximum value in each column
        maxArr = np.array(maxArr)       # turn the list into an np array

        return maxArr

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        dataRange = [self.min(headers, rows), self.max(headers, rows)] # use min and max functions to return list of ranges
        return dataRange

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        
        sumData = self.data.select_data(headers, rows) # grab the data to be summed up
        n = sumData.shape[0] # store the number of samples in the data
        sum = np.sum(sumData, axis=0) # sum all of the entries
        mean = sum/n # calculate the mean
        return mean


    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        meanData = self.data.select_data(headers, rows) # grab data for the mean
        n = meanData.shape[0] # store the number of samples in the data
        mean = self.mean(headers, rows) # calculate the mean
        oneArr = np.ones(meanData.shape) # create an array of ones that is same shape as meanData
        meanArr = (mean * oneArr) # mulitply mean into oneArr to create array of means
        diffArr = meanData - mean # subtract mean from mean data to calculate the difference, stored in array
        sqArr = np.square(diffArr) # square each element in the difference array
        sumArr = np.sum(sqArr, axis=0) # sum each entry in array
        var = np.divide(sumArr, n-1) # calculate the variance
        return var

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        return np.sqrt(self.var(headers,rows)) # calculate sqaure root of variance to find standard deviaton

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()  # show the plot

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        headers = [ind_var, dep_var]    # get list of independent and dependent variables as headers
        self.data.set_headers(headers)  # set self.headers to headers passed
        rows = []   # empty list to select all rows

        plotData = self.data.select_data(headers, rows) # grab data for plots
        x = plotData[:,0]   # store data from all rows in first column in x
        y = plotData[:,1]   # store data from all rows in second column in y 

        plt.figure(figsize=(6,5))
        plt.scatter(plotData[:,0], plotData[:,1])    # plot scatter plot with x and y
        plt.title(title)    # create title with title passed
        plt.xlabel(ind_var) # set x label to ind_var
        plt.ylabel(dep_var) # set y label to dep_var

        return x, y


    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        fig, axes = plt.subplots(len(data_vars), len(data_vars), sharex='col', sharey='row', figsize = fig_sz) # create subplots

        fig.set_figwidth(fig_sz[0]) # set width
        fig.set_figheight(fig_sz[1])    # set height

        plotData = self.data.select_data(data_vars) # select the appropriate data from data_vars

        for i in range(len(data_vars)): # begin nested loop to iterate through all possible variable combinations
            
            for j in range(len(data_vars)):
                
                axes[i,j].scatter(plotData[:,j], plotData[:,i], marker='.') # create a scatter plot of the data

                if i==len(data_vars)-1: # if the plot is on the bottom row, add a label to y axis

                    axes[i,j].set_xlabel(data_vars[j])

                if j==0:    # if the plot is on the first column, add a label to x axis
                    
                    axes[i,j].set_ylabel(data_vars[i])

        plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0.5, hspace=0.5)   # adjust spacing of subplots

        plt.tight_layout(pad = 2.0)

        return fig, axes