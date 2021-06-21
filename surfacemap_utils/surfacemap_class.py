import numpy as np
from scipy.interpolate import RegularGridInterpolator
# documentation
import typing
from typing import Tuple, Iterable
#

class surfacemap():
    def __init__( self, s_array:Iterable[float], \
                        t_array:Iterable[float], \
                        datamatrix: Tuple[Tuple[Tuple[float,...],...],...],\
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
        #raise Exception( "".join(f"{p}: {self._interp(p)}\n" \
        #        for p in ((0.51,0.51),(0.49,0.51),(0.49,0.49),(0.51,0.49))),\
        #        datamatrix[13:16,13:16]
        #        )
        grad_datamatrix = np.gradient( datamatrix, axis=(0,1) )
        grad_datamatrix[0] = grad_datamatrix[0] * ( len(s_array)-1)
        grad_datamatrix[1] = grad_datamatrix[1] * ( len(t_array)-1)
        ds_interp = RegularGridInterpolator( (s_array, t_array), \
                                            grad_datamatrix[0], \
                                            bounds_error=bounds_error )
        dt_interp = RegularGridInterpolator( (s_array, t_array), \
                                            grad_datamatrix[1], \
                                            bounds_error=bounds_error )
        perpendicular = np.cross( grad_datamatrix[0], grad_datamatrix[1] )
        perp_length = np.linalg.norm( perpendicular, axis=2)
        perpendicular_norm = perpendicular \
                            / perp_length.reshape((*perp_length.shape,1))
        self._perpendicular_interp =RegularGridInterpolator((s_array, t_array),\
                                            perpendicular, \
                                            bounds_error=bounds_error )
        self._interp_ds, self._interp_dt = ds_interp, dt_interp
        #grad_datamatrix = np.gradient( datamatrix, axis=(0,1) )
        dxyz_to_dst = np.zeros( (datamatrix.shape[0], datamatrix.shape[1],\
                                datamatrix.shape[2],2) )
        dxyz_to_dst[:,:,:,0] = grad_datamatrix[0]
        dxyz_to_dst[:,:,:,1] = grad_datamatrix[1]
        #dxyz_to_dst[:,:,:,2] = perpendicular
        dxyz_to_dst2 = np.linalg.pinv( dxyz_to_dst )
        #raise Exception(  dxyz_to_dst2[10,10] @ grad_datamatrix[1][10,10], s_array[10] )
        self._invgrad_interp =RegularGridInterpolator((s_array, t_array),\
                                            dxyz_to_dst2, \
                                            bounds_error=bounds_error )
        #raise Exception( self.get_inverse_gradmatrix(*p) @ self.get_derivate_to_st(*p)[0] )
        #raise Exception( self.get_derivate_to_st( 0.27792966, 0.36340138 ) )

    def get_inverse_gradmatrix( self, x:float, y:float ) \
                                            ->Tuple[Tuple[float,...]]:
        return self._invgrad_interp((x,y))

    def get_value_to_st( self, x:float, y:float ) -> Tuple[float,...]:
        return self._interp(( x, y ))

    def get_perpendicular_to_st( self, x:float, y:float ) -> Tuple[float,...]:
        return self._perpendicular_interp(( x, y ))

    def get_derivate_to_st( self, x:float, y:float ) \
                            -> Tuple[Tuple[float,...],Tuple[float,...]]:
        """
        :return: return len(tuple)==2 is derivate in the 
                two direction dx,dy at position x, y
        """
        return self._interp_ds( (x, y) ), self._interp_dt( (x, y) )


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

