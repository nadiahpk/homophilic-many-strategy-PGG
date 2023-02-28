# general use functions for plotting dynamics of four-variable games
# functions with a net_ prefix plot dynamics on the surfaces of the tetrahedron
# and functions with a tet_ prefix are for 3D plots
#
# modified from tetraplot to deal with the function coming in as a class

import numpy as np

import itertools as it
import matplotlib.path as mpltPath

import scipy.optimize
from math import isclose
from scipy.integrate import solve_ivp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# general purpose coordinates on the net

def net_xy2bary(tris, xy, idx_omit=None):

    if idx_omit is None:

        # find which index is omitted by figuring out which triangle we're in
        for idx_omit, trij in enumerate(tris):

            path = mpltPath.Path(trij)
            inside = path.contains_point(xy)

            if inside:
                break

    # make the conversion from xy to bary
    trij = tris[idx_omit]
    r3 = trij[2, :]
    M = trij.T[:2, :2] - np.tile(r3, (2, 1)).T      # conversions matrix
    ls_2 = np.linalg.inv(M) @ (xy - np.array(r3)).T # bary coordinates for first 2
    ls_3 = np.append(ls_2, 1-sum(ls_2))             # third coord

    # insert the zero coordinate that was omitted
    ls = np.concatenate( (ls_3[:idx_omit], np.array([0]), ls_3[idx_omit:]) )

    return ls


def net_bary2xy(tris, ls, idx_omit=None):

    # the barycentric coords should sum to 1
    #assert(sum(ls) == 1)

    if len(ls) == 4:

        # we've been given all four coordinates,
        # which means we may have to figure out which variable was omitted

        if idx_omit is None:

            # find the first variable omitted
            check0 = list(ls==0)
            assert any(check0)
            idx_omit = check0.index(True)

        # the 3 coordinates
        lsj = np.concatenate((ls[:idx_omit], ls[idx_omit+1:]))

    elif len(ls) == 3:

        lsj = ls

    # the triangle is the one that omits that index
    trij = tris[idx_omit]
    xy = trij.T @ lsj

    return xy


# plotting specific to the net

def net_get_face_mesh(ngrid=32):
    '''
    Gets the mesh points in barycentric coordinates for a tetrahedron face.


    Inputs
    ---

    ngrid, int
        Each axes is split into ngrid divisions. The default is 
        32 = 2^5, which means the axes are subdivided 5 times.


    Outputs
    ---

    lV, list of 1x3 numpy arrays of floats
        A list of grid points in the mesh in barycentric coordinates.

    '''

    nface = 3
    lV = [ np.array([outcome.count(s) 
        for s in range(nface)])/ngrid 
        for outcome in it.combinations_with_replacement(range(nface), ngrid) ]

    return lV


def net_plot_initialise(ax, strat_names=['A', 'B', 'C', 'D']):
    '''
    Draws the boundaries of the net plot with labelled vertices, and 
    returns the list 'tris' of the x,y coordinates of the triangle 
    vertices.


    Inputs
    ---

    ax, AxesSubplot
        The plotting object returned by e.g., ax = plt.figure().add_subplot(1,1,1)

    strat_names, list of four strings
        The labels of the axes to plot. The first three are the 
        vertices of the central triangle in the net, and the fourth is
        the final vertex of the tetrahedron, which will be placed on 
        the outer vertices of the net. 


    Outputs
    ---

    tris, list of four 3x2 np arrays
        Defines the x,y coordinates of the four triangles of the net 
        in order of the axis omitted (i.e., omit 1st axis, 2nd axis)
    '''

    # label the coordinates
    # strat_names = ['A', 'B', 'C', 'D']

    # corners of the centre triangle
    r1 = np.array([0, 0])               # A
    r2 = np.array([1, 0])               # B
    r3 = np.array([1/2, np.sqrt(3)/2])  # C

    # extra corners for element D
    r4_lft = np.array([-1/2, np.sqrt(3)/2])     # top left corner
    r4_rgt = np.array([ 3/2, np.sqrt(3)/2])     # top right corner
    r4_bot = np.array([ 1/2, -np.sqrt(3)/2])    # bottom corner

    # label the triangles 1 to 4 from left to right and top to bottom
    # and define the corner order in variable order
    tri1 = np.array(np.matrix([r1, r3, r4_lft]))    # A, C, D
    tri2 = np.array(np.matrix([r1, r2, r3]))        # A, B, C
    tri3 = np.array(np.matrix([r2, r3, r4_rgt]))    # B, C, D
    tri4 = np.array(np.matrix([r1, r2, r4_bot]))    # A, B, D

    # each triangle above omits the follow variable:
    # tri1 omits 1, tri2 omits 3, tri3 omits 0, tri4 omits 2
    # so order the triangles by the variable they omit
    tris = [tri3, tri1, tri4, tri2]

    # lines to draw - outer and inner triangle
    lines = [ np.array(np.matrix([r4_lft, r4_rgt, r4_bot, r4_lft])), np.array(np.matrix([r1, r2, r3, r1])) ]
    for line in lines:
        ax.plot(line[:,0], line[:,1], color='black', alpha=0.5, lw=0.5)

    # draw labels
    for r, strat_name in zip([r1, r2, r3, r4_lft, r4_rgt, r4_bot], strat_names + [strat_names[-1], strat_names[-1]]):

        xi = r[0]
        yi = r[1]

        if yi > 0:
            yi += 0.1
        else:
            yi -= 0.1

        if xi > 0.5:
            xi += 0.1
        elif xi < 0.5:
            xi -= 0.1

        text = ax.text(xi, yi, strat_name, ha='center', va='center', fontsize='x-large')

    # other plot settings
    ax.axis('equal')
    ax.axis('off')
    margin=0.05
    ax.set_ylim(ymin = -np.sqrt(3)/2 - margin, ymax = np.sqrt(3)/2 + margin)
    ax.set_xlim(xmin = -0.5 - margin, xmax = 1.5 + margin)

    return tris

def net_plot_fixed_points(ax, tris, fp_baryV):

    for idx_omit, fp_bary in zip(range(4), fp_baryV):

        # convert fixed points from barycentric to xyz and plot
        fp_xy = [ net_bary2xy(tris, ls, idx_omit) for ls in fp_bary ]

        # plot steady states
        if fp_xy:
            xV, yV = zip(*fp_xy)
            ax.scatter(xV, yV, color='black')


def net_plot_fixed_points_w_stability(ax, tris, fp_baryV, stab_fnc, abs_tol=1e-6):

    # check stability each point

    fp_stabilityV = list()
    for idx_omit, fp_barys in zip(range(4), fp_baryV):

        fp_stabilitys = list()
        for fp_bary_ in fp_barys:

            # zero out any that should be zero
            fp_bary = [0 if isclose(v, 0, abs_tol=abs_tol) else v for v in fp_bary_]

            if fp_bary.count(0) == 2: # can't find Jacobian

                fp_stabilitys.append('NA')

            else:

                # put back the missing strategy and check stability
                fp_bary4 = fp_bary[:idx_omit] + [0] + fp_bary[idx_omit:]
                fp_stabilitys.append(stab_fnc(fp_bary4))

        fp_stabilityV.append(fp_stabilitys)


    # add points to plot with colour marking stability

    stab2colour = {'neutral': 'white', 'stable': 'blue', 'unstable': 'red', 'NA': 'black'}

    for idx_omit, fp_barys, fp_stabilitys in zip(range(4), fp_baryV, fp_stabilityV):

        # convert fixed points from barycentric to xyz and plot
        fp_xys = [net_bary2xy(tris, ls, idx_omit) for ls in fp_barys]

        # plot steady states
        for (x, y), stab in zip(fp_xys, fp_stabilitys):
            ax.scatter(x, y, color=stab2colour[stab])


def net_plot_dynamics_rate(ax, tris, lV, pvalsV):

    for idx_omit, pvals, in zip(range(4), pvalsV):

        xyV = [ net_bary2xy(tris, ls, idx_omit) for ls in lV ]
        xV, yV = zip(*xyV)

        # plot colour contour
        ax.tricontourf(xV, yV, pvals, alpha=0.8, cmap='viridis')


def net_plot_dynamics_dirn(ax, tris, lV, dirn_normV):

    for idx_omit, dirn_norm in zip(range(4), dirn_normV):

        xyV = [ net_bary2xy(tris, ls, idx_omit) for ls in lV ]
        xV, yV = zip(*xyV)

        # plot quivers
        ax.quiver(xV, yV, dirn_norm.T[0], dirn_norm.T[1], angles='xy', width=0.0013, scale=90, pivot='mid')

# dynamics on the net

def net_find_fixed_points(lV, vdot_fnc):
    '''
    Starting at each point in lV, find a fixed, and return a list of unique fixed points found


    Inputs
    ---

    lV, list of 1x3 numpy arrays of floats
        A list of grid points in the mesh in barycentric coordinates.

    vdot_fnc, function
        A function that returns the selection grad given three 
        barycentric coords and the index of the omitted variable, 
        i.e., def vdot_fnc(v, idx_omit=None)


    Outputs
    ---

    fp_baryV, list of 4 lists of 1x3 numpy arrays of floats
        Barycentric coordinates of the fixed points found. Each of 
        the 4 lists corresponds to one face of the tetrahedron. They 
        are in order of the variable omitted order (i.e., omit 1st, 
        2nd, 3rd, 4th).
    '''

    fp_baryV = list() # a place to store lists of fixed points for each tetrahedron surface
    for idx_omit in range(4): # for each tetrahedron surface

        fp_bary = []  # a place to store the roots found
        delta = 1e-12 # small error permitted to check if roots found are within [0,1]

        for ls in lV: # for each point on the surface (in barycentric coordinates)

            fp_try = np.array([])

            # try to find a root
            sol = scipy.optimize.root(lambda ls: vdot_fnc(ls, idx_omit=idx_omit), ls, method="hybr")

            if sol.success:

                fp_try = sol.x

                # check if the solution found was in the simplex

                # check if the coordinate sum is approx 1
                if not isclose(np.sum(fp_try), 1., abs_tol=2.e-3):
                    continue

                # check if each coordinate value was approx within [0,1]
                if not np.all((fp_try>-delta) & (fp_try <1+delta)):
                    continue

            else:

                continue

            # if it passed the checks above, add if it's new and really a fixed point
            if not np.array([np.allclose(fp_try, x, atol=1e-7) for x in fp_bary]).any():
                if np.allclose(vdot_fnc(fp_try, idx_omit = idx_omit), 0, atol=1e-7):
                    fp_bary.append(fp_try)

        # store
        fp_baryV.append(fp_bary)

    '''
    # NOTE - could be useful later
    # remove repeated fixed points across idx_omits
    cleaned_fp_baryV = [fp_baryV[0]]
    for idx_omit in range(1,4):

        fp_bary = fp_baryV[idx_omit]
        idxs = [ i for i in range(4) if i != idx_omit ]

        cleaned_fp_bary = list() # a place to store new ones
        for fp in fp_bary:

            found_match = False

            for cf_idx_omit in range(idx_omit): # only compare to previous ones

                if not isclose(fp[idxs.index(cf_idx_omit)], 0, abs_tol=1e-6):
                    # if it doesn't have a zero for the cf_idx_omit element,
                    # then can't be in that group 
                    continue

                # create a version of fp that can be compared directly to elements of cf_idx
                re_fp = list(fp)
                re_fp = re_fp[:idx_omit] + [0] + re_fp[idx_omit:] # add the omitted idx back
                re_fp = np.array(re_fp[:cf_idx_omit] + re_fp[cf_idx_omit+1:]) # omit the cf_idx_omit

                for cf_fp in cleaned_fp_baryV[cf_idx_omit]:

                    if np.allclose(re_fp, cf_fp, atol=1e-6):
                        found_match = True
                        break

                if found_match is True:
                    continue # no need to check other cf_idxs

            if found_match is False:
                cleaned_fp_bary.append(fp)

        cleaned_fp_baryV.append(cleaned_fp_bary)

    return cleaned_fp_baryV
    '''
    return fp_baryV


def net_calc_grad_on_mesh(tris, lV, vdot_fnc):
    '''
    Calculate the selection gradient direction and strength at each point in the mesh lV


    Inputs
    ---

    tris, list of four 3x2 np arrays
        Defines the x,y coordinates of the four triangles of the net 
        in order of the axis omitted (i.e., omit 1st axis, 2nd axis)

    lV, list of 1x3 numpy arrays of floats
        A list of grid points in the mesh in barycentric coordinates.

    vdot_fnc, function
        A function that returns the selection grad given three 
        barycentric coords and the index of the omitted variable, 
        i.e., def vdot_fnc(v, idx_omit=None)
    '''

    # places to store lists of gradient info for each tetrahedron surface
    strengthsV = list()
    dirn_normV = list()
    for idx_omit in range(4): # for each tetrahedron surface

        dirn_bary = [ vdot_fnc(ls, idx_omit=idx_omit) for ls in lV ]
        dirn_xy = [ net_bary2xy(tris, ls, idx_omit) for ls in dirn_bary ]
        strengths = [ np.linalg.norm(xy) for xy in dirn_xy ]
        dirn_norm = np.array([ xy/p if p > 0 else np.array([0, 0]) for xy, p in zip(dirn_xy, strengths) ])

        # store
        strengthsV.append(strengths)
        dirn_normV.append(dirn_norm)

    return (strengthsV, dirn_normV)

# coordinates in the tetrahedron

def tet_bary2xyz(rs, ls):
    '''
    rs = np.array(np.matrix([r1, r2, r3, r4]))
    ls = np.array([l1, l2, l3, l4]) barycentric coords
    '''

    xyz = rs.T @ ls

    return xyz

def tet_xyz2bary(rs, v):
    '''
    rs = np.array(np.matrix([r1, r2, r3, r4]))
    v = np.array([x, y, z]) cartesian coordinates
    '''

    r4 = rs[3,:]
    M = rs.T[:3,:3] - np.tile(r4, (3,1)).T          # conversions matrix
    ls_3 = np.linalg.inv(M) @ (v - np.array(r4)).T  # bary coordinates for first three
    ls = np.append(ls_3, 1-sum(ls_3))                 # final coord

    return ls

# general purpose functions for the tetrahedral plot

def tet_get_mesh(ngrid=16):
    '''
    Gets the mesh points in barycentric coordinates for a whole tetrahedron


    Inputs
    ---

    ngrid, int
        Each axes is split into ngrid divisions. The default is 
        16 = 2^4, which means the axes are subdivided 4 times.


    Outputs
    ---

    lV, list of 1x3 numpy arrays of floats
        A list of grid points in the mesh in barycentric coordinates.

    '''

    # the 4 is because 4 axes
    lV = [ np.array([outcome.count(s) 
        for s in range(4)])/ngrid 
        for outcome in it.combinations_with_replacement(range(4), ngrid) ]

    return lV

def tet_plot_initialise(ax, strat_names=['A', 'B', 'C', 'D']):
    '''
    Draws the boundaries of the tetrahedral plot with labelled 
    vertices, and returns the list 'vertexs' of the x,y coordinates of the 
    tetrahedron vertices.


    Inputs
    ---

    ax, AxesSubplot
        The plotting object returned by e.g., ax = plt.figure().add_subplot(projection='3d')

    strat_names, list of four strings
        The labels of the axes to plot. The first three are the 
        vertices of the central triangle in the net, and the fourth is
        the final vertex of the tetrahedron, which will be placed on 
        the outer vertices of the net. 

    elev, float
        Elevation of the plot view, used in ax.view_init(elev=15)

    azim, float
        Azimuth of the plot view, used in ax.view_init(azim=-65)


    Outputs
    ---

    vertexs, 3x4 numpy matrix of floats
        Defines the x,y,z coordinates of the vertices of the 
        tetrahedron
    '''

    # the vertices of a regular tetrahedron, from https://en.wikipedia.org/wiki/Tetrahedron
    r1 = np.array((np.sqrt(8/9), 0, -1/3))
    r2 = np.array((-np.sqrt(2/9), np.sqrt(2/3), -1/3))
    r3 = np.array((-np.sqrt(2/9), -np.sqrt(2/3), -1/3))
    r4 = np.array((0, 0, 1))
    vertexs = np.array(np.matrix([r1, r2, r3, r4]))


    # draw each line of the tetrahedron
    lines = [ [r1, r2], [r2, r3], [r3, r1], [r1, r4], [r2, r4], [r3, r4] ]
    for line in lines:
        rr1, rr2 = line
        x, y, z = zip(rr1, rr2)
        ax.plot(x, y, z, color='black', lw=0.5)

    # label the vertices
    for r, strat_name, i in zip(vertexs, strat_names, range(4)):

        xi = r[0]
        yi = r[1]
        zi = r[2]

        if zi < 0:
            zi -= 0.1
        else:
            zi += 0.1

        text = ax.text(xi, yi, zi, strat_name, ha='center', va='center', fontsize='x-large')

        # make the C label faded a bit so we know it's in the background
        if i == 2:
            text.set_alpha(.4)

    # plot aesthetics
    ax.set_axis_off()
    ax.view_init(elev=15, azim=55)
    ax.set_zlim3d((-0.45,0.65))
    ax.set_xlim3d((-0.70,0.40))
    ax.set_ylim3d((-0.65,0.45))

    return vertexs

def tet_plot_fixed_points(ax, vertexs, fp_bary):

    # convert list of bary coordinates to xyz coords
    xyzV = [ list(tet_bary2xyz(vertexs, ls)) for ls in fp_bary ]
    xV, yV, zV = zip(*xyzV)

    # plot them
    ax.scatter(xV, yV, zV, color='black')

def tet_draw_arrowheads(ax, xbeg, xend, ybeg, yend, zbeg, zend, colour='blue', alpha=1):

    # copied from https://stackoverflow.com/a/22867877
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    a = Arrow3D([xbeg, xend], [ybeg, yend], [zbeg, zend],
            mutation_scale=20, lw=0, arrowstyle="-|>", color=colour, alpha=alpha)
    ax.add_artist(a)

# dynamics in the tetrahedron

def tet_find_fixed_points(lV, vdot_fnc):
    '''
    Starting at each point in lV, find a fixed, and return a list of 
    unique fixed points found.


    Inputs
    ---

    lV, list of 1x3 numpy arrays of floats
        A list of grid points in the mesh in barycentric coordinates.

    vdot_fnc, function
        A function that returns the selection grad given three 
        barycentric coords and the index of the omitted variable, 
        i.e., def vdot_fnc(v, idx_omit=None)


    Outputs
    ---

    fp_bary, list of 1x3 numpy arrays of floats
        Barycentric coordinates of the fixed points found. 
    '''

    fp_bary = []  # a place to store the roots found
    delta = 1e-12 # small error permitted to check if roots found are within [0,1]

    for ls in lV: # for each point on the surface (in barycentric coordinates)

        fp_try = np.array([])

        # try to find a root
        sol = scipy.optimize.root(vdot_fnc, ls, method="hybr")

        if sol.success:

            fp_try = sol.x

            # sanity check all derivatives are indeed near zero at this solution
            if max(np.abs(vdot_fnc(fp_try))) > 1e-7:
                continue

            # check if the solution found was in the simplex

            # check if the coordinate sum is approx 1
            if not isclose(np.sum(fp_try), 1., abs_tol=2.e-3):
                continue

            # check if each coordinate value was approx within [0,1]
            if not np.all((fp_try>-delta) & (fp_try <1+delta)):
                continue

        else:
            continue

        # if it passed the checks above, add the root to the list if it's new
        if not np.array([np.allclose(fp_try, x, atol=1e-7) for x in fp_bary]).any():

            fp_bary.append(fp_try)

    return fp_bary

def get_trajectory(vertexs, v0, tspan, vdot_fnc):
    '''
    Returns a tractory in xV, yV, zV coords
    '''

    traj_bary = get_trajectory_bary(vertexs, v0, tspan, vdot_fnc)

    # convert solution to xyz coords
    xV, yV, zV = zip(*[ list(tet_bary2xyz(vertexs, np.append(ls, 1-sum(ls)))) for ls in traj_bary ])

    return (xV, yV, zV)

def get_trajectory_bary(vertexs, v0, tspan, vdot_fnc):
    '''
    Returns a tractory in barycentric coords
    '''

    v0 = v0[:3]
    method = 'LSODA'

    fnc = lambda t, v: vdot_fnc(np.array([v[0], v[1], v[2], 1-sum(v)]))[:3]
    sol = solve_ivp(fnc, tspan, v0, method=method)

    return (sol.y.T)
