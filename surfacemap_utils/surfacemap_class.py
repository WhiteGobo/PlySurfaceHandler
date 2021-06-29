import numpy as np
from scipy.interpolate import RegularGridInterpolator
# documentation
import typing
from typing import Tuple, Iterable
#

class surfacemap():
    def __init__( self, s_array:Iterable[float], \
                        t_array:Iterable[float], \
                        datamatrix: Tuple[Tuple[Tuple[float,...],...],...], \
                        bounds_error = False):
        """
        :todo: replace in grad_matr /len(starr) with something like /grad(sarr)
        """
        self.s_array, self.t_array = s_array, t_array
        self.datamatrix = datamatrix
        myinterpolator = RegularGridInterpolator( (s_array, t_array), \
                                                    datamatrix, \
                                                    bounds_error=bounds_error )
        self._interp = myinterpolator
        grad_datamatrix = np.gradient( datamatrix, axis=(0,1) )
        grad_datamatrix[0] = grad_datamatrix[0] * ( len(s_array)-1)
        grad_datamatrix[1] = grad_datamatrix[1] * ( len(t_array)-1)
        ds_interp = RegularGridInterpolator( (s_array, t_array), \
                                            grad_datamatrix[0], \
                                            bounds_error=bounds_error )
        dt_interp = RegularGridInterpolator( (s_array, t_array), \
                                            grad_datamatrix[1], \
                                            bounds_error=bounds_error )
        self._interp_ds, self._interp_dt = ds_interp, dt_interp

    def _get_uplength( self ):
        upnodes = np.array( self.datamatrix[:,-1] )
        lengtharray = np.linalg.norm( upnodes[:-1] - upnodes[1:], axis=-1 )
        distances = [ np.sum( lengtharray[:i] ) \
                        for i in range(len(lengtharray)+1)]
        return distances
    uplength = property( fget=_get_uplength )
    def _get_leftlength( self ):
        upnodes = np.array( self.datamatrix[0,:] )
        lengtharray = np.linalg.norm( upnodes[:-1] - upnodes[1:], axis=-1 )
        distances = [ np.sum( lengtharray[:i] ) \
                        for i in range(len(lengtharray)+1)]
        return distances
    leftlength = property( fget=_get_leftlength )
    def _get_downlength( self ):
        upnodes = np.array( self.datamatrix[:,0] )
        lengtharray = np.linalg.norm( upnodes[:-1] - upnodes[1:], axis=-1 )
        distances = [ np.sum( lengtharray[:i] ) \
                        for i in range(len(lengtharray)+1)]
        return distances
    downlength = property( fget=_get_downlength )
    def _get_rightlength( self ):
        upnodes = np.array( self.datamatrix[-1,:] )
        lengtharray = np.linalg.norm( upnodes[:-1] - upnodes[1:], axis=-1 )
        distances = [ np.sum( lengtharray[:i] ) \
                        for i in range(len(lengtharray)+1)]
        return distances
    rightlength = property( fget=_get_rightlength )

    def get_point_matrix( self ):
        if are_equidistant( self.s_array, self.t_array ):
            return self.datamatrix
        else:
            xyzst_matrix = np.zeros( (len(self.s_array),len(self.t_array), \
                                        2+self.datamatrix.shape[-1] ) )
            xyzst_matrix[ :, :, :-2 ] = self.datamatrix
            xyzst_matrix[ :, :, -2: ] = np.meshgrid( s_array, t_array )
            return xyzst_matrix

    def get_value_to_st( self, x:float, y:float ) -> Tuple[float,...]:
        return self._interp(( x, y ))

    def get_derivate_to_st( self, x:float, y:float ) \
                            -> Tuple[Tuple[float,...],Tuple[float,...]]:
        """
        :return: return len(tuple)==2 is derivate in the 
                two direction dx,dy at position x, y
        """
        return self._interp_ds( (x, y) ), self._interp_dt( (x, y) )

    def get_transmatrix_dxyz_to_dst( self, x:float, y:float ):
        tmp = self.get_derivate_to_st( x, y )
        return np.linalg.pinv( tmp )

    @classmethod
    def from_datamatrix( cls, \
                        datamatrix: Tuple[Tuple[Tuple[float,...],...],...], \
                        ):
        shape = np.array( datamatrix ).shape
        s_array = np.linspace( 0, 1, shape[0] )
        t_array = np.linspace( 0, 1, shape[1] )
        return cls( s_array, t_array, datamatrix )

    @classmethod
    def from_arrays( cls, shape :Tuple[int,int], *arrays: Iterable[float] ):
        if not all( shape[0]*shape[1] == len(arr) for arr in arrays ):
            raise TypeError( "arrays must have the same length "\
                                                    "shape[0]*shape[1]" )
        width_array = np.linspace( 0, 1, shape[0] )
        height_array = np.linspace( 0, 1, shape[1] )
        datamap = np.ndarray( ( *shape, len(arrays) ) )
        for i, arr in enumerate( arrays ):
            datamap[ :, :, i ] = np.array( arr ).reshape( shape )
        return cls( width_array, height_array, datamap )

    @classmethod
    def from_arrays_with_nonregular_xyarrays( cls, x_array: Iterable[float], \
                                                    y_array: Iterable[float], \
                                                    *arrays: Iterable[float] ):
        shape = ( len( x_array ), len( y_array ) )
        if not all( shape[0]*shape[1] == len(arr) for arr in arrays ):
            raise TypeError( "arrays must have the same length "\
                                                    "shape[0]*shape[1]" )
        datamap = np.ndarray( ( *shape, len(arrays) ) )
        for i, arr in enumerate( arrays ):
            datamap[ :, :, i ] = np.array( arr ).reshape( shape )
        return cls( x_array, y_array, datamap )

    def visualize_with_matplotlib( self ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure()
        ax = fig.gca( projection='3d' )
        surf = ax.plot_wireframe( *(self.datamatrix[:,:,i] for i in (0,1,2) ) )

        #x_test = np.linspace(0,1)
        #y_test = np.linspace(0,3)
        #z_test = myinter( (x_test, y_test) )
        #ax.plot( x_test, y_test, z_test )
        plt.show()

def are_equidistant( *arrays ):
    for arr in arrays:
        arr = np.array( arr )
        d = arr[0] - arr[1]
        if not np.allclose( d, arr[1:-1] - arr[2:] ):
            return False
    return True
