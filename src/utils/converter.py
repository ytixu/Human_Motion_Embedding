import numpy as np
import math
import copy

#################################
####### Data conversions ########
## quad, expmap, rotmap, euler ##
#################################
# these are all copied form
# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py

def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  bad_quad = False
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    print (np.abs(np.linalg.norm(q)-1), "quat2expmap: input quaternion is not norm 1")
    bad_quad = True
    #raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r, bad_quad

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul

def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
  Args
  R: a 3x3 rotation matrix
  Returns
  eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul

#################################
####### Data conversions ########
####### euler -> rotmap #########
#################################
# these are all copied form
# https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py

def euler2mat(ai, aj, ak, axes='sxyz'):

  """Return rotation matrix from Euler angles and axis sequence.
  Parameters
  ----------
  ai : float
  First rotation angle (according to `axes`).
  aj : float
  Second rotation angle (according to `axes`).
  ak : float
  Third rotation angle (according to `axes`).
  axes : str, optional
  Axis specification; one of 24 axis sequences as string or encoded
  tuple - e.g. ``sxyz`` (the default).
  Returns
  -------
  mat : array-like shape (3, 3) or (4, 4)
  Rotation matrix or affine.
  Examples
  --------
  >>> R = euler2mat(1, 2, 3, 'syxz')
  >>> np.allclose(np.sum(R[0]), -1.34786452)
  True
  >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
  >>> np.allclose(np.sum(R[0]), -0.383436184)
  True
  """
  _AXES2TUPLE = {
  'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
  'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
  'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
  'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
  'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
  'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
  'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
  'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

  _NEXT_AXIS = [1, 2, 0, 1]

  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i+parity]
  k = _NEXT_AXIS[i-parity+1]

  if frame:
    ai, ak = ak, ai
  if parity:
    ai, aj, ak = -ai, -aj, -ak

  si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
  ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
  cc, cs = ci*ck, ci*sk
  sc, ss = si*ck, si*sk

  M = np.eye(3)
  if repetition:
    M[i, i] = cj
    M[i, j] = sj*si
    M[i, k] = sj*ci
    M[j, i] = sj*sk
    M[j, j] = -cj*ss+cc
    M[j, k] = -cj*cs-sc
    M[k, i] = -sj*ck
    M[k, j] = cj*sc+cs
    M[k, k] = cj*cc-ss
  else:
    M[i, i] = cj*ck
    M[i, j] = sj*sc-cs
    M[i, k] = sj*cc+ss
    M[j, i] = cj*sk
    M[j, j] = sj*ss+cc
    M[j, k] = sj*cs-sc
    M[k, i] = -sj
    M[k, j] = cj*si
    M[k, k] = cj*ci
  return M

def euler2rotmap(euler):
  return euler2mat(euler[2], euler[1], euler[0])

#################################
####### Foward kinematics #######
#################################
# these are all copied form
# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py
# modified revert_coordinate_space()

def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 99

  # Structure that indicates parents for each joint
  njoints   = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    r = angles[ expmapInd[i] ]

    thisRotation = expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]


  return np.reshape( xyz, [-1] )

def revert_coordinate_space(channels, R0, T0, mat_type='expmap'):
  """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m
  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
    mat_type: parameterization of the matrix (added by ytixu)
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  # Loop through the passed posses
  for ii in range(n):
    #### modified here
    func = expmap2rotmat
    if mat_type == 'euler':
      func = euler2rotmap

    R_diff = func( channels[ii, rootRotInd] )
    #### end
    R = R_diff.dot( R_prev )

    #### modified here
    if mat_type == 'euler':
      channels_rec[ii, rootRotInd] = rotmat2euler(R)
    else:
    ####end
      channels_rec[ii, rootRotInd], bad_quad = rotmat2expmap(R)
      if bad_quad:
        print ii

    T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree
  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                    17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

  offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
  offset = offset.reshape(-1,3)

  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55],
            [],
            [59, 60, 58],
            [],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76],
            []]

  expmapInd = np.split(np.arange(4,100)-1,32)

  return parent, offset, rotInd, expmapInd

#################################
############# END ###############
#################################

def sequence_expmap2euler(data):
  '''
  Convert a sequence of exponential map to euler angle
  '''
  for i in np.arange(data.shape[0]):
    for k in np.arange(3,97,3):
      data[i,k:k+3] = rotmat2euler( expmap2rotmat( data[i,k:k+3] ))
  return data

def sequence_expmap2quater(data):
  '''
  Convert a sequence of exponential map to quaternion
  '''
  new_data = np.zeros(data.shape[0], 33*4)
  for i in np.arange(data.shape[0]):
    for j,k in enumate(np.arange(3,97,3)):
      new_data[i,j*4:(j+1)*4] = rotmat2quat( expmap2rotmat( data[i,k:k+3] ))
  return new_data

def sequence_quater2expmap(data):
  '''
  Convert a sequence of quaternion to exponential map
  '''
  new_data = np.zeros(data.shape[0], 99)
  for i in np.arange(data.shape[0]):
    for j,k in enumate(np.arange(3,97,3)):
      new_data[i,k:k+3] = quat2expmap( data[i,j*4:(j+1)*4] )
  return new_data


def sequence_quater2euler(data):
  return sequence_expmap2euler(sequence_quater2expmap(data))

def sequence_something2xyz__(data, data_type='expmap'):
  '''
  Convert a sequence of exponential map or euler to euclidean space using FK
  Borrowed from
  https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L156
  '''
  data[:,:6] = 0 # remove global rotation and translation

  nframes, _ = data.shape
  xyz = np.zeros((nframes, 96))

  # Put them together and revert the coordinate space
  data = revert_coordinate_space( data, np.eye(3), np.zeros(3), data_type )

  # Compute 3d points for each frame
  parent, offset, rotInd, expmapInd = _some_variables()
  for i in range( nframes ):
    xyz[i,:] = fkl( data[i,:], parent, offset, rotInd, expmapInd )

  return xyz

def sequence_expmap2xyz(data):
  return sequence_something2xyz__(data)

def sequence_euler2xyz(data):
  return sequence_something2xyz__(data, 'euler')

def sequence_quater2xyz(data):
  data = sequence_quater2expmap(data)
  return sequence_something2xyz__(data)

# For plotting poses
# relevant_coords = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

def animate(xyz):
  '''
  Animate motion in euclidean
  Borrowed from
  https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L156
  '''
  import viz
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob = viz.Ax3DPose(ax)

  for i in range(xyz.shape[0]):
    ob.update( xyz[i,:] )
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.002)
