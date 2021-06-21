from scipy.sparse import lil_matrix, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.sparse.linalg import spsolve
import itertools
import numpy as np
from scipy.optimize import minimize, Bounds
from .surfacemap_class import surfacemap
# documentation
from typing import Tuple, Iterable, Callable, List
from numpy.typing import ArrayLike
#


def create_gridmap_from( up, left, down, right, \
                    vertexpositions:Iterable[Tuple[float,float,float]], edges ):
    """
    :todo: instead of linspace use all used u and v positions as crossproduct
    """
    number_vertices = len( vertexpositions )
    edges = set( frozenset(e) for e in edges )
    edges = list( tuple(e) for e in edges )
    # rough estimation of uv-pos of vertices

    u_array, v_array = _estimate_uv_from_edgegrid( number_vertices, \
                                        edges, vertexpositions, \
                                        up, left, down, right )
    borderindices = set(itertools.chain(up,left,down,right))

    # minimize the discrepancy between edgelength of real-pos and uv-pos
    # This optimize seems to be not necessary and not working
    #startparams, params_to_uv_arrays, get_delta = _create_optimiser( \
    #                                    vertexpositions, edges, borderindices,\
    #                                    u_array, v_array )
    #foundparams = minimize( get_delta, startparams , method='nelder-mead',\
    #                                    options={'xatol':1e-8, 'disp':True})
    #u_array, v_array = params_to_uv_arrays( np.array(foundparams.x) )

    # generate uv to position generator from verticeposition
    uv_to_xyz = generate_surfaceinterpolator( vertexpositions, u_array, v_array)
    l, l2 = 21, 23
    #todo: instead of linspace use all used u and v positions as crossproduct
    s_array, t_array = np.linspace(0,1,l), np.linspace(0,1,l2)
    datamatrix = np.zeros((l,l2,3))
    for i, s in enumerate( s_array ):
        for j, t in enumerate( t_array ):
            datamatrix[i,j,:] = uv_to_xyz( s, t )
    firstsurfmap = surfacemap( s_array, t_array, datamatrix )
    firstsurfmap.visualize_with_matplotlib()

    # generate grid and minimize deviation of real distances of gridvertices
    ulength, vlength = 20,20
    grid_u, grid_v = np.meshgrid( np.linspace(0,1,vlength), \
                                    np.linspace(0,1,ulength) )
    grid_xyz = _optimise_grid_position( grid_u, grid_v, \
                                firstsurfmap.get_value_to_st, \
                                firstsurfmap.get_derivate_to_st, \
                                firstsurfmap.get_perpendicular_to_st, \
                                firstsurfmap.get_inverse_gradmatrix )

    secondsurfmap = surfacemap( np.linspace(0,1,ulength), \
                                np.linspace(0,1,vlength), grid_xyz )
    secondsurfmap.visualize_with_matplotlib()


def _optimise_grid_position( grid_u, grid_v, uv_to_xyz, uv_to_dxyz, \
                            uv_to_perpendicular, uv_to_matrix_xyztouv ):
    u_length, v_length = grid_u.shape
    innerindex_grid = itertools.product(range(1,u_length-1),range(1,v_length-1))
    v_offset = (u_length-2) * (v_length-2)
    grid_indices_to_param_uv_indices = { pos:(index, index+v_offset) \
                                for index, pos in enumerate( innerindex_grid )}
    startparams = np.zeros( ( 2*v_offset, ) )
    for p, param_indices in grid_indices_to_param_uv_indices.items():
        ui, vi = param_indices
        startparams[ui] = grid_u[ p ]
        startparams[vi] = grid_v[ p ]
    def params_to_grid_uv( params ):
        new_grid_u, new_grid_v = np.array( grid_u ), np.array( grid_v )
        for p, param_indices in grid_indices_to_param_indices.items():
            ui, vi = param_indices
            new_grid_u[ p ] = params[ ui ]
            new_grid_v[ p ] = params[ vi ]
        return new_grid_u, new_grid_v
    startpos = np.zeros( ( u_length, v_length, 3 ) )
    for i,j  in itertools.product( range(u_length), range(v_length) ):
        startpos[ i,j,: ] = uv_to_xyz( grid_u[i,j], grid_v[i,j] )

    def params_to_grid_perpendicular( params ):
        current_dxyz = np.zeros( (u_length-2, v_length-2, 3) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            tmp = np.array(uv_to_dxyz( params[ui], params[vi] ))
            tmp2 = np.cross( tmp[0], tmp[1] )
            current_dxyz[ p[0]-1, p[1]-1,:]  = tmp2 / np.linalg.norm(tmp2)
        return current_dxyz
    def params_to_grid_xyz( params ):
        current_pos = np.array( startpos )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            current_pos[ p[0], p[1],: ] = uv_to_xyz( params[ui], params[vi] )
        return current_pos
    def params_to_grid_dst_dxyz( params ):
        current_dxyz = np.zeros( (2, u_length-2, v_length-2, 3) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            current_dxyz[ :,p[0]-1, p[1]-1,:] \
                    = np.array(uv_to_dxyz( params[ui], params[vi] )).reshape((2,3))
        return current_dxyz
    def params_to_matrix_xyz_to_st( params ):
        current_dxyz = np.zeros( (u_length-2, v_length-2, 2, 3) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            tmp = np.array(uv_to_dxyz( params[ui], params[vi] )).T
            current_dxyz[ p[0]-1, p[1]-1,:,:]  = np.linalg.pinv( tmp )
        return current_dxyz
        current_dxyz = np.zeros( (u_length-2, v_length-2, 2,3) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            current_dxyz[ p[0]-2, p[1]-2,:,:] \
                    = np.array(uv_to_matrix_xyztouv( params[ui], params[vi] ))
        return current_dxyz


    def get_jacobian( params ):
        xyz = params_to_grid_xyz( params )
        forceshape = np.subtract( xyz.shape, (2,2,0) )
        force = np.zeros( forceshape )
        forcewindow = np.array(((0,1,0),(1,-4,1),(0,1,0)))
        for i in range(3):
            force[:,:,i] = convolve( xyz[:,:,i],forcewindow,mode="valid")

        invmatrix = params_to_matrix_xyz_to_st( params )
        tmp = invmatrix@(-force.reshape((force.shape[0], force.shape[1], 3, 1)))
        tmp_s = tmp[:,:,0,0]
        tmp_t = tmp[:,:,1,0]
        jacobian = np.zeros( (len(params),) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            jacobian[ ui ] = tmp_s[ p[0]-1, p[1]-1 ]
            jacobian[ vi ] = tmp_t[ p[0]-1, p[1]-1 ]
        return jacobian
        raise Exception( force.shape, invmatrix.shape , \
                        (invmatrix@force.reshape((force.shape[0],force.shape[1],3,1))),\
                        tmp_s.shape, tmp_t.shape)

        q = np.linalg.norm(force, axis=2)
        forcedirection = force# / q.reshape( (*q.shape, 1) )
        dxyz = params_to_grid_dst_dxyz( params )
        tmp_s = -4 * dxyz[0] * forcedirection
        tmp_t = -4 * dxyz[1] * forcedirection
        tmp_s[ :  , 1:  ] += dxyz[0,  :  ,  :-1] * forcedirection[:,1:]
        tmp_s[ :  ,  :-1] += dxyz[0,  :  , 1:  ] * forcedirection[:,:-1]
        tmp_s[1:  ,  :  ] += dxyz[0,  :-1,  :  ] * forcedirection[1:,:]
        tmp_s[ :-1,  :  ] += dxyz[0, 1:  ,  :  ] * forcedirection[:-1,:]
        tmp_t[ :  , 1:  ] += dxyz[1,  :  ,  :-1] * forcedirection[:,1:]
        tmp_t[ :  ,  :-1] += dxyz[1,  :  , 1:  ] * forcedirection[:,:-1]
        tmp_t[1:  ,  :  ] += dxyz[1,  :-1,  :  ] * forcedirection[1:,:]
        tmp_t[ :-1,  :  ] += dxyz[1, 1:  ,  :  ] * forcedirection[:-1,:]

        tmp_s = np.sum( tmp_s, axis=2)
        tmp_t = np.sum( tmp_t, axis=2)
        #print( "hello" )
        #print( tmp_s )
        #print( tmp_t )
        #print( (force.reshape((8,8,1,3)) @ invmatrix) )
        m = force.reshape((force.shape[0], force.shape[1],1,3)) @ invmatrix
        tmp_s = -m[:,:,0,0]
        tmp_t = -m[:,:,0,2]
        #tmp_s = sum( tmp_s[:,:,i] for i in range(3) )
        #tmp_t = sum( tmp_t[:,:,i] for i in range(3) )
        jacobian = np.zeros( (len(params),) )
        print( "xyz\n", xyz )
        print( "f", force )
        print( "inv", invmatrix )
        print( "m", m )
        raise Exception( xyz, force, invmatrix, m)
        raise Exception( xyz, params, tmp_s, tmp_t )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            jacobian[ ui ] = tmp_s[ p[0]-1, p[1]-1 ]
            jacobian[ vi ] = tmp_t[ p[0]-1, p[1]-1 ]
        return jacobian


    from scipy.signal import convolve
    def foo_to_minimize( params ):
        xyz = params_to_grid_xyz( params )
        forceshape = np.subtract( xyz.shape, (2,2,0) )
        force = np.zeros( forceshape )
        forcewindow = np.array(((0,1,0),(1,-4,1),(0,1,0)))
        for i in range(3):
            force[:,:,i] = convolve( xyz[:,:,i],forcewindow,mode="valid")
        perpendicular = params_to_grid_perpendicular( params )
        perp_force_scalar = ( \
                force.reshape(forceshape[0], forceshape[1],1,3) \
                @ perpendicular.reshape(forceshape[0], forceshape[1],3,1) \
                ).reshape( (*forceshape[:2],1) )
        nonperp_force = force - perpendicular * perp_force_scalar
        q = np.sum( np.linalg.norm(nonperp_force, axis=2))
        return q

        q = np.sum( np.linalg.norm(force, axis=2) )
        return q
        return np.sum( np.linalg.norm(force, axis=2))
    mybounds = [(0,1)]*len(startparams)
    foundparams = minimize( foo_to_minimize, startparams , \
                                        #method='BFGS',\
                                        jac = get_jacobian, \
                                        bounds=mybounds, \
                                        options={'gtol':1e-8, 'disp':True,\
                                        #'maxfev':200*len(startparams),\
                                        #'maxiter':200*len(startparams), \
                                        })
    #foundparams = minimize( foo_to_minimize, \
    #                                    startparams , method='POWELL',\
    #                                    jac = get_jacobian, \
    #                                    #bounds = mybounds, \
    #                                    #options={'xatol':1e-8, 'disp':True,\
    #                                    options={'gtol':1e-8, 'disp':True,\
    #                                    'maxfev':200*len(startparams),\
    #                                    'maxiter':200*len(startparams), })
    print( foundparams )
    grid_xyz = params_to_grid_xyz( foundparams.x )
    #grid_xyz = params_to_grid_xyz( startparams )
    return grid_xyz


def _edgethingis_for_grid( u_length, v_length, grid_indices_to_param_indices ):
    number_params = len( grid_indices_to_param_indices )
    u_edges = np.array([ [( (u1,v), (u2,v) ) \
                for u1, u2 in zip( range(u_length-1), range(1,u_length) )] \
                for v in range(1, v_length-1) ])
    v_edges = np.array([ [( (u,v1), (u,v2) ) \
                for v1, v2 in zip( range(v_length-1), range(1,v_length) )] \
                for u in range(1, u_length-1) ])

    def vertpos_to_sum_squarelength_in_each_row( grid_xyz ) -> float:
        sum_lengthsquare = 0.0
        for row in itertools.chain( u_edges, v_edges ):
             sum_lengthsquare \
                     += np.sum( np.square( [ \
                        np.linalg.norm( grid_xyz[ v1[0], v1[1],: ] \
                                - grid_xyz[ v2[0], v2[1],: ] ) \
                        for v1, v2 in row ] ))
        return sum_lengthsquare

    def vertpos_to_sum_squarelengthderivation_in_each_row( grid_xyz ) -> float:
        length_array_of_u_edges = np.zeros( u_edges.shape )
        length_array_of_v_edges = np.zeros( v_edges.shape )
        sum_lengthdifference_in_path = 0.0
        for row in itertools.chain( u_edges, v_edges ):
             sum_lengthdifference_in_path \
                     += np.sum( np.square( np.gradient( [ \
                        np.linalg.norm( grid_xyz[ v1[0], v1[1],: ] \
                                - grid_xyz[ v2[0], v2[1],: ] ) \
                        for v1, v2 in row ] )))
        return sum_lengthdifference_in_path
    return vertpos_to_sum_squarelengthderivation_in_each_row,\
            vertpos_to_sum_squarelength_in_each_row

    mat_vert_to_uedges = lil_matrix( (numbervertices, paramlength) )
    for v1, v2 in u_edges:
        mat_vert_to_uedges[ gridtrans_1d[ v1 ], gridtrans_params[ v2 ]] = 1
        mat_vert_to_uedges[ gridtrans_1d[ v2 ], gridtrans_params[ v2 ]] += -1
        mat_vert_to_uedges[ gridtrans_1d[ v2 ], gridtrans_params[ v1 ]] = 1
        mat_vert_to_uedges[ gridtrans_1d[ v1 ], gridtrans_params[ v1 ]] += -1
    mat_vert_to_uedges = mat_vert_to_uedges.tocsr()
    mat_vert_to_vedges = lil_matrix( (numbervertices, paramlength) )
    for v1, v2 in v_edges:
        mat_vert_to_vedges[ gridtrans_1d[ v1 ], gridtrans_params[ v2 ]] = 1
        mat_vert_to_vedges[ gridtrans_1d[ v2 ], gridtrans_params[ v2 ]] += -1
        mat_vert_to_vedges[ gridtrans_1d[ v2 ], gridtrans_params[ v1 ]] = 1
        mat_vert_to_vedges[ gridtrans_1d[ v1 ], gridtrans_params[ v1 ]] += -1
    mat_vert_to_vedges = mat_vert_to_vedges.tocsr()
    def get_paramtominimize( x_pos_1d, y_pos_1d, z_pos_1d ):
        delta_x_uedges = x_pos_1d * mat_vert_to_uedges
        delta_y_uedges = y_pos_1d * mat_vert_to_uedges
        delta_z_uedges = z_pos_1d * mat_vert_to_uedges
        delta_x_vedges = x_pos_1d * mat_vert_to_vedges
        delta_y_vedges = y_pos_1d * mat_vert_to_vedges
        delta_z_vedges = z_pos_1d * mat_vert_to_vedges
        du_arrays = (delta_x_uedges, delta_x_uedges, delta_x_uedges )
        delta_uedges = np.sqrt( np.sum( np.square(arr) for arr in du_arrays ))
        dv_arrays = (delta_x_vedges, delta_x_vedges, delta_x_vedges )
        delta_vedges = np.sqrt( np.sum( np.square(arr) for arr in dv_arrays ))
        return np.sum( delta_uedges ) + np.sum( delta_vedges )
    return get_paramtominimize


    #mat = dok_matrix( (vertexnumbers, vertexnumbers-len(borderindices)) )
    #filtered_vertexnumbers = vi for vi in range(vertexnumbers) \
    #                            if vi not in borderindices
    #for i, vertex_index in enumerate( filtered_vertexnumbers ):
    #    mat[ vertex_index, i ] = 1
    #add_matrix = mat.transpose().tocsr()



def _create_function_params_grid_translator( grid_u, grid_v ):
    startgrid_u, startgrid_v = np.array( grid_u ), np.array( grid_v )
    shape = grid_u.shape
    paramhalflength = (shape[0]-2) * (shape[1]-2)
    def grids_to_params( grid_u, grid_v ):
        params = np.zeros((2 * paramhalflength,))
        params[:paramhalflength] = grid_u[ 1:-1, 1:-1 ] \
                                    .reshape((paramhalflength,))
        params[paramhalflength:] = grid_v[ 1:-1, 1:-1 ] \
                                    .reshape((paramhalflength,))
        return params
    def params_to_grids( params ):
        grid_u = np.array( startgrid_u )
        grid_v = np.array( startgrid_v )
        grid_u[ 1:-1, 1:-1 ] = params[ :paramhalflength ]\
                                .reshape(((shape[0]-2), (shape[1]-2)))
        grid_v[ 1:-1, 1:-1 ] = params[ paramhalflength: ]\
                                .reshape(((shape[0]-2), (shape[1]-2)))
        return grid_u, grid_v
    def param_to_positionarrays( params ):
        grid_u, grid_v = params_to_grids( params )
        x_array = np.zeros((vertexnumbers,))
        y_array = np.zeros((vertexnumbers,))
        z_array = np.zeros((vertexnumbers,))
        for arraypos, gridpos in arraypos_to_gridpos.items():
            x_array[ arraypos ], y_array[ arraypos ], z_array[ arraypos ] \
                                = uv_to_xyz( (grid_u[gridpos], grid_v[gridpos]))
        return x_array, y_array, z_array
    return grids_to_params, params_to_grids, param_to_positionarrays


def generate_surfaceinterpolator( vertexpositions, u_array, v_array ):
    st_coordinates = np.ndarray( ( len(vertexpositions), 2 ) )
    st_coordinates[ :, 0 ] = u_array
    st_coordinates[ :, 1 ] = v_array
    delaunay_triang = Delaunay( st_coordinates )
    #xyz_as_uvmap = CloughTocher2DInterpolator( delaunay_triang,vertexpositions)
    uv_to_xyz = LinearNDInterpolator( delaunay_triang, vertexpositions)
    return uv_to_xyz


def _estimate_uv_positions( up, left, down, right, vertexpositions, edges ):
    number_vertices = len( vertexpositions )
    interaction_matrix = lil_matrix((number_vertices, number_vertices))
    startpositions_u, startpositions_v = _create_initcondition_for_ub_finding()
    def array_to_uvpos( uv_array ):
        return uv_array.reshape((number_vertices,2))
    def uvpos_to_array( uv_positions ):
        return uv_positions.reshape((2 * number_vertices,))


def _create_optimiser( vertexpositions, edges, borderindices, start_u, start_v):
    real_edgelength, get_uv_edgelength, uv_arrays_to_force = _create_edgeutils(\
                                            edges, vertexpositions )
    vertexnumbers = len( vertexpositions )
    halfparamnumber = (vertexnumbers - len( borderindices ))
    paramnumbers = 2 * halfparamnumber
    def param_uvarrays_to_array1d( u_array, v_array ) -> ArrayLike:
        return np.array((u_array, v_array)).reshape( paramnumbers )
    def param_array1d_to_uvarrays( uv1d_array ) -> Tuple[ ArrayLike, ArrayLike]:
        return uv1d_array.reshape((2, halfparamnumber))

    mat = dok_matrix( (vertexnumbers, vertexnumbers-len(borderindices)) )
    filtered_vertexnumbers = [ vi for vi in range(vertexnumbers) \
                                if vi not in borderindices ]
    for i, vertex_index in enumerate( filtered_vertexnumbers ):
        mat[ vertex_index, i ] = 1
    add_matrix = mat.transpose().tocsr()
    remove_matrix = mat.tocsr()
    border_u_array = np.array( [u if i in borderindices else 0 \
                                for i, u in enumerate( start_u )] )
    border_v_array = np.array( [v if i in borderindices else 0 \
                                for i, v in enumerate( start_v )] )
    #@np.vectorize
    def removebordervertices_from_singlearray( array ):
        return array * remove_matrix
    #@np.vectorize
    def addbordervertices_uvpositions( u_array, v_array ):
        u = border_u_array + u_array * add_matrix
        v = border_v_array + v_array * add_matrix
        return u, v

    def get_delta( current_params ):
        u_array, v_array = param_array1d_to_uvarrays( current_params )
        u_array, v_array = addbordervertices_uvpositions( u_array, v_array )
        force_u, force_v = uv_arrays_to_force( u_array, v_array )
        return np.sum( np.sqrt( np.square( u_array ) + np.square( v_array ) ))

    def params_to_uv_arrays( current_params ):
        #u_array, v_array = param_array1d_to_uvarrays( current_params )
        u_array, v_array = current_params.reshape((2, halfparamnumber))
        u_array, v_array = addbordervertices_uvpositions( u_array, v_array )
        return u_array, v_array

    startparams = param_uvarrays_to_array1d( \
                removebordervertices_from_singlearray( start_u ), \
                removebordervertices_from_singlearray( start_v ), \
                )

    return startparams, params_to_uv_arrays, get_delta


def _create_edgeutils( edges, vertexpositions:List[Tuple[float,...]] ) \
                                    -> Tuple[Iterable[float], Callable]:
    number_edges= len( edges )
    number_vertices = len( vertexpositions )
    real_edgelength = np.ndarray((number_edges,))
    for i, edge in enumerate( edges ):
        a,b = edge
        real_edgelength[i] = np.linalg.norm( np.subtract( vertexpositions[a],
                                                        vertexpositions[b] ))
    def get_uv_edgelength( u_array, v_array ):
        edgelength = np.array( \
                np.linalg.norm( (u_array[a]-u_array[b],v_array[a]-v_array[b]))\
                for i, a, b in enumerate( edges )
                )
        return edgelength

    p_to_edgedings = dok_matrix((number_vertices, number_edges))
    for i, edge in enumerate( edges ):
        p_to_edgedings[ edge[0], i ] = 1
        p_to_edgedings[ edge[1], i ] = -1
    edgedings_to_p = p_to_edgedings.transpose()
    edgedings_to_p = edgedings_to_p.tocsr()
    p_to_edgedings = p_to_edgedings.tocsr()
    def uv_arrays_to_force( u_array, v_array ):
        uvelength = get_uv_edgelength( u_array, v_array )
        edge_u = u_array * p_to_edgedings
        edge_v = v_array * p_to_edgedings
        delta_u_array = edge_u * edgedings_to_p
        delta_v_array = edge_v * edgedings_to_p
        return delta_u_array, delta_v_array
    return real_edgelength, get_uv_edgelength, uv_arrays_to_force


def _estimate_uv_from_edgegrid( number_vertices, edges, vertexpositions, \
                                            up, left, down, right ):
    randnodes_indices = set( itertools.chain( up, left, down, right ))
    interaction_matrix = _create_spring_interaction_matrix( edges, \
                                            randnodes_indices, number_vertices,\
                                            vertexpositions)
    startpositions_u, startpositions_v = _create_initcondition_for_uv_finding(\
                                            number_vertices, vertexpositions, \
                                            up, left, down, right )
    u_array = spsolve( interaction_matrix, startpositions_u )
    v_array = spsolve( interaction_matrix, startpositions_v )
    return u_array, v_array


def _create_spring_interaction_matrix( edges, border_indices, number_vertices, \
                                        vertexpositions ):
    #edges = set( frozenset(e) for e in edges )
    #edges = sorted( sorted(e) for e in edges )
    interaction_matrix = lil_matrix( (number_vertices, number_vertices) )
    edgelength_from = lambda a, b: np.linalg.norm( \
                        np.subtract(vertexpositions[a], vertexpositions[b]))
    for a, b in edges:
        inverselength = 1/edgelength_from(a, b)
        interaction_matrix[ a, b ] = inverselength
        interaction_matrix[ b, a ] = inverselength
        interaction_matrix[ a, a ] += -inverselength
        interaction_matrix[ b, b ] += -inverselength
    for i in border_indices:
        for j in range( number_vertices ):
            interaction_matrix[ i, j ] = 0
        interaction_matrix[ i, i ] = 1
    return interaction_matrix.tocsr()
    

def _calc_pathdistance_array( vertexpositions, path_indices ):
    path_distance = [ 0.0 ]
    for a, b in zip( path_indices[:-1], path_indices[1:] ):
        p1 = vertexpositions[ a ]
        p2 = vertexpositions[ b ]
        path_distance.append( np.linalg.norm( np.subtract(p1,p2) ) )
    return np.array( path_distance )


def _create_initcondition_for_uv_finding( number_vertices, vertexpositions, \
                                                up, left, down, right ):
    startpositions_u = lil_matrix((number_vertices,1))
    startpositions_v = lil_matrix((number_vertices,1))

    up_distance = _calc_pathdistance_array( vertexpositions, up )
    up_distance = up_distance / up_distance[-1]
    for index, d in zip( up, up_distance ):
        startpositions_u[index] = d
        startpositions_v[index] = 1.0
    left_distance = _calc_pathdistance_array( vertexpositions, left )
    left_distance = left_distance / left_distance[-1]
    for index, d in zip( left, left_distance ):
        startpositions_u[index] = 1.0
        startpositions_v[index] = 1-d
    down_distance = _calc_pathdistance_array( vertexpositions, down )
    down_distance = down_distance / down_distance[-1]
    for index, d in zip( down, down_distance ):
        startpositions_u[index] = 1-d
        startpositions_v[index] = 0.0
    right_distance = _calc_pathdistance_array( vertexpositions, right )
    right_distance = right_distance / right_distance[-1]
    for index, d in zip( right, right_distance ):
        startpositions_u[index] = 0.0
        startpositions_v[index] = d
    return startpositions_u.tocsr(), startpositions_v.tocsr()


def helpinghand( datamatrix ):
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
