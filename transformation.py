'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Derek Hessinger
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data



class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`
        '''
        super().__init__(data)
        self.orig = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.
        '''

        new_data = self.orig.select_data(headers)   # select data from headers
        dictionary = {} # create dictionary to hold header2col indices

        for idx, header in enumerate(headers): # loop through each header/index in headers
            dictionary[header] = idx    # map the index to header

        self.data = data.Data(data=new_data,headers=headers,header2col=dictionary) # create new data object with relevant info

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.
        '''

        ones = np.ones((self.data.get_num_samples(), 1))    # create np array of ones
        newArr = np.hstack((self.data.data, ones))  # stack array onto data to add homogeneous column
        return newArr

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''

        dims = self.data.get_num_dims() # get the number of variables
        matrix = np.eye(dims+1, dims+1) # create a matrix to size (m+1, m+1)

        for i in range(len(magnitudes)):    # loop through magnitudes
            matrix[i,dims] = magnitudes[i]  # append matrix to insert translation values
        return matrix

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        dims = self.data.get_num_dims() # get the number of variables
        matrix = np.eye(dims+1, dims+1) # create a matrix to size (m+1, m+1)

        for i in range(len(magnitudes)): # loop through magnitudes
            matrix[i,i] = magnitudes[i] # append matrix to add scale values

        return matrix

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
        
        tMatrix = self.translation_matrix(magnitudes) # make transformation matrix
        headers = self.data.get_headers()   # get headers

        ogData = self.get_data_homogeneous()    # add homogeneous row to data
        ogData = ogData.T   # transpose data
        result = (tMatrix@ogData).T # compute matrix multiplication
        result = result[:, 0:-1]    # remove homogeneous row

        self.data = data.Data(data=result,headers=headers,header2col=self.data.get_mappings()) # create new data object with relevant info

        return result
        
        

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''

        sMatrix = self.scale_matrix(magnitudes) # make transformation matrix
        headers = self.data.get_headers()   # get headers

        ogData = self.get_data_homogeneous()    # add homogeneous row to data
        ogData = ogData.T   # transpose data
        result = (sMatrix@ogData).T # compute matrix multiplication
        result = result[:, 0:-1]    # remove homogeneous row

        self.data = data.Data(data=result,headers=headers,header2col=self.data.get_mappings()) # create new data object with relevant info

        return result
        

    def transform(self, C): #TODO: fix float values in result
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`
        '''

        headers = self.data.get_headers()   # get headers

        cMatrix = C
        ogData = self.get_data_homogeneous()    # add homogeneous row to data
        ogData = ogData.T   # transpose data
        result = (cMatrix@ogData).T # compute matrix multiplication
        result = result[:, 0:-1]    # remove homogeneous row

        self.data = data.Data(data=result,headers=headers,header2col=self.data.get_mappings()) # create new data object with relevant info

        return result

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        dims = self.data.get_num_dims() # get dims of data
        headers = self.data.get_headers()

        minArr = self.min(headers) # find global min
        gMin = min(minArr)
        maxArr = self.min(headers) # find global max
        gMax = max(maxArr)
        diff = gMax - gMin # find difference

        tMag = [] # magnitude for translation
        sMag = [] # magnitude for scale

        for i in range(dims):   # for each variable

            tMag.append(-gMin)  # create list of global min
            sMag.append(1/diff) # create list of 1/diff

        transMatrix = self.translation_matrix(tMag) # create translation matrix with translation magnitudes
        scaleMatrix = self.scale_matrix(sMag)   # create scale matrix with scale magnitudes

        cMatrix = transMatrix@scaleMatrix   # create transformation matrix

        result = self.transform(cMatrix)   # normalize the matrix
        
        self.data = data.Data(headers=headers, data=result, header2col=self.data.get_mappings())

        return result

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        headers = self.data.get_headers() # get headers
        min = self.min(headers) # find min of each variable column
        max = self.max(headers) # find max of each variable coloumn
        diff = max-min

        tMag = []
        sMag = []

        for i in range(len(headers)):  
            tMag.append(-min[i])    # add minimums to list
            sMag.append(1/diff[i])     # add diffs to list

        transMatrix = self.translation_matrix(tMag) # create translation matrix
        scaleMatrix = self.scale_matrix(sMag)   # create scale matrix

        cMatrix = transMatrix@scaleMatrix # compute transformation matrix

        result = self.transform(cMatrix)

        self.data = data.Data(headers=headers, data=result, header2col=self.data.get_mappings())
        
        return result

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        homog = self.get_data_homogeneous() # get a homogeneous array of data
        rotate = np.eye(homog.shape[1], homog.shape[1]) # create an identity matrix for rotation matrix of correct size

        radians = degrees * (np.pi/180) # compute the degrees in radians

        headers = self.data.get_headers()   # get headers from projected data

        if headers.index(header) == 0:  # if the data variable is the "x" coordinate, create rotation matrix about x
            rotate[1,1] = np.cos(radians)
            rotate[1,2] = -(np.sin(radians))
            rotate[2,1] = np.sin(radians)
            rotate[2,2] = np.cos(radians)
        elif headers.index(header) == 1:    # if the data variable is the "y" coordinate, create rotation matrix about y
            rotate[0,0] = np.cos(radians)
            rotate[0,2] = np.sin(radians)
            rotate[2,0] = -(np.sin(radians))
            rotate[2,2] = np.cos(radians)
        elif headers.index(header) == 2:    # if the data variable is the "z" coordinate, create rotation matrix about z
            rotate[0,0] = np.cos(radians)
            rotate[0,1] = -(np.sin(radians))
            rotate[1,0] = np.sin(radians)
            rotate[1,1] = np.cos(radians)
        
        return rotate

    def rotate_3d(self, header, degrees):   #TODO: fix long float returned
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        rMatrix = self.rotation_matrix_3d(header, degrees)  # create rotation matrix

        rotated = self.transform(rMatrix)   # transform data with rotation matrix

        self.data = data.Data(headers=self.data.get_headers(), data=rotated, header2col=self.data.get_mappings())   # reassign values
        
        return rotated

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        headers = [ind_var, dep_var, c_var]    # get list of independent and dependent variables as headers
        self.data.set_headers(headers)  # set self.headers to headers passed
        rows = []   # empty list to select all rows

        plotData = self.data.select_data(headers, rows) # grab data for plots

        x = plotData[:,0]   # store data from ind_var in x
        y = plotData[:,1]   # store data from dep_var in y
        z = plotData[:,2]   # store data from c_var in z

        color_map = palettable.colorbrewer.sequential.Oranges_9
        plt.scatter(x, y, c=z, cmap=color_map.mpl_colormap, edgecolor='black')    # plot scatter plot with x, y, and z

        if title != None:
            plt.title(title)    # create title with title passed

        plt.xlabel(ind_var) # set x label to ind_var
        plt.ylabel(dep_var) # set y label to dep_var
        bar = plt.colorbar()
        bar.set_label(c_var)

