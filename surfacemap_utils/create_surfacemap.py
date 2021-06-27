from scipy.sparse import lil_matrix, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve
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
    roughgridshape = (40, 40)
    ulength, vlength = 15, 15
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
    #todo: instead of linspace use all used u and v positions as crossproduct
    s_array = np.linspace(0,1,roughgridshape[0])
    t_array = np.linspace(0,1,roughgridshape[1])
    datamatrix = np.zeros((*roughgridshape,3))
    for i, s in enumerate( s_array ):
        for j, t in enumerate( t_array ):
            datamatrix[i,j,:] = uv_to_xyz( s, t )
    firstsurfmap = surfacemap( s_array, t_array, datamatrix )

    # generate grid and minimize deviation of real distances of gridvertices
    grid_u, grid_v = np.meshgrid( np.linspace(0,1,vlength), \
                                    np.linspace(0,1,ulength) )
    grid_xyz = _optimise_grid_position( grid_u, grid_v, \
                                firstsurfmap.get_value_to_st, \
                                firstsurfmap.get_derivate_to_st, \
                                firstsurfmap.get_transmatrix_dxyz_to_dst, \
                                )

    secondsurfmap = surfacemap( np.linspace(0,1,ulength), \
                                np.linspace(0,1,vlength), grid_xyz )
    return secondsurfmap


def _optimise_grid_position( grid_u, grid_v, uv_to_xyz, uv_to_dxyz, \
                                            transmatrix_dxyz_to_duv ):
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
    def params_to_matrix_xyz_to_st( params ):
        current_dxyz = np.zeros( (u_length-2, v_length-2, 2, 3) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            current_dxyz[ p[0]-1, p[1]-1,:,:]  \
                    = transmatrix_dxyz_to_duv( params[ui], params[vi] ).T
        return current_dxyz

    forceshape = np.subtract( (*grid_u.shape, 3), (2,2,0) )
    shape_force_for_transformation = ( forceshape[0], forceshape[1], 3, 1 )
    forcewindow = np.array(((0,1,0),(1,-4,1),(0,1,0)))
    def get_jacobian( params ):
        xyz = params_to_grid_xyz( params )
        force = np.zeros( forceshape )
        for i in range(3):
            force[:,:,i] = convolve( xyz[:,:,i], forcewindow, mode="valid" )
        invmatrix = params_to_matrix_xyz_to_st( params )
        tmp = invmatrix@(-force.reshape( shape_force_for_transformation ))
        tmp_s = tmp[:,:,0,0]
        tmp_t = tmp[:,:,1,0]
        jacobian = np.zeros( (len(params),) )
        for p, param_indices in grid_indices_to_param_uv_indices.items():
            ui, vi = param_indices
            jacobian[ ui ] = tmp_s[ p[0]-1, p[1]-1 ]
            jacobian[ vi ] = tmp_t[ p[0]-1, p[1]-1 ]
        return jacobian

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
    mybounds = [(0,1)]*len(startparams)
    foundparams = minimize( foo_to_minimize, startparams , \
                                        #method='BFGS',\
                                        jac = get_jacobian, \
                                        bounds=mybounds, \
                                        options={'gtol':1e-8, 'disp':False,\
                                        })
    grid_xyz = params_to_grid_xyz( foundparams.x )
    return grid_xyz


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
    path_distance = np.array( path_distance )
    path_distance = [ sum( path_distance[:i+1] ) \
                        for i in range(path_distance.shape[0])]
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

