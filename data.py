'''data.py
Reads CSV files, stores data, access/filter data by variable name
Derek Hessinger
CS 251 Data Analysis and Visualization
Spring 2023
'''
import csv
import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0'''
        
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col

        if self.filepath != None:   # if filepath is there, call read
            self.read(filepath)

        return

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned'''

        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)  # create reader object

            self.filepath = filepath    # set file path
            line = 0                    # initialize line
            colNum = []                 # create column set
            self.header2col = {}        # create header2col dictionary to store headers and corresponding column nums
            self.data = []              # create data set to hold data entries
            list = []                   # create list set to hold headers

            for row in reader:
                if line == 0:   # For the first row (headers)
                    list = row  # take the headers from the first row and store in a set
                    line+=1 # increment to next row

                elif line == 1: # grab header datatype and indices
                    if "numeric" not in row:   # check to ensure csv file has data type row and raise error if not
                        raise Exception("FileError: CSV file is missing data type row. Must contain data type row.")
                    idx = self.rowIndices(row)  # use helper function to get row indices
                    self.headers = self.headerIndices(idx, list) # set header and corresponding indices with helper function
                    line+=1     # increment to next row

                else:           # for remaining rows, add data points to newRow set, then add newRows to data set
                    dataRow = [] # create empty set for data rows
                    for i in idx:   # for each element in each column, add data from row into set
                        dataRow.append(float(row[i]))
                    self.data.append(dataRow)   # add new set to data

            self.mapHeaders() # use helper function to make a dictionary of cols to headers

            self.data = np.array(self.data) # create np array of data
        return

    def headerIndices(self, contents, arr):
        '''Helper function that returns headers within a set
        Parameters:
        ---------
        contents: python list of header indices
        arr: numpy array
        Returns :
        ---------
        list of header indices'''
        headers = []    # create set to hold headers
        for i in contents:   # for each index, add corresponding header to set
            x = arr[i].rstrip() # remove whitespace
            headers.append(x)   # add string to headers
        return headers

    def rowIndices(self, row):
        '''Helper function that returns row indices of numeric data
        Parameters:
        -----------
        row: python list of headers to extract data from
        
        Returns:
        --------
        Python list of row indices with numeric entries'''
        indices = []    # create empty set to hold indices
        for i in range(len(row)):   # for the number of headers in the row
            if row[i] == 'numeric': # if the header is a numeric element
                indices.append(i)   # add to set
        return indices

    def mapHeaders(self):
        '''Helper function that maps headers to their corresponding indices in data object'''
        for i in range(len(self.headers)):  # for the number of headers
            self.header2col[self.headers[i]] = i    # cast the header index to the header
        return

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        result = str(self.filepath) + " (" + str(self.data.shape[0]) + "x" + str(self.data.shape[1]) + ") \n"   # add filepath and shape to string
        result += "Headers: \n" 
        for header in self.headers: # add headers to string
            result += header + "    "

        result += "\n-------------------------------\n"

        for i in range(5):  # add first five rows to string
            for data in self.data[i]:
                result += str(data) + "     "
            result += "\n"
            
            if i >= (np.ma.size(self.data, axis = 0)) -1:   # if i is greater than 5, break elements are not printed further
                break

        return result

    def get_headers(self):
       '''Get method for headers

        Returns:
        -----------
        Python list of str.
        ''' 
       return self.headers

    def set_headers(self, headers):
        '''Method to set headers'''
        self.headers = headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col


    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.headers)
    
    
    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        ''' 
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        indices = []
        for head in headers:
            indices.append(self.header2col[head])

        return indices

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''

        copy = np.copy(self.data)
        return copy

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        samples = []    # list to hold data samples
        dataRange = self.get_num_samples()  # get total number of samples in data

        for i in range(5):  # iterate through first 5 samples
            if i > dataRange - 1: # account for small data
                break
            samples.append(self.get_sample(i))  # add to samples
        
        return np.array(samples)

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        samples = []    # list to hold data samples
        dataRange = self.get_num_samples()  # get total sample size

        if dataRange >= 5:  # for larger datasets
            num = self.get_num_samples() - 5
            for i in range(5):
                if i > dataRange - 1:
                    break
                samples.append(self.get_sample(num+i))
        
        else:   # for smaller datasets
            for i in range(dataRange):
                if i > dataRange - 1:
                    break
                samples.append(self.get_sample(i))

        return np.array(samples)

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row: end_row]
        return

    def select_data(self, headers, rows=[]): 
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''

        dataSamples = []    # list to hold data samples
        headerIdx = self.get_header_indices(headers)    # header indices

        if len(rows) > 0:   # add specific rows if greater than 0
            dataSamples.append(self.data[np.ix_(rows, headerIdx)])
        
        else:   # add all rows otherwise
            dataSamples.append(self.data[np.ix_(np.arange(self.data.shape[0]), headerIdx)])
        
        dataSamples = np.array(dataSamples) # convert to np array
        dataSamples = np.squeeze(dataSamples, axis = 0) # squeeze dimensions
        return dataSamples