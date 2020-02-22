import numpy as np

from wavedata.tools.core import geometry_utils


class VoxelGrid2D(object):
    """
    Voxel grids represent occupancy info. The voxelize_2d method projects a point cloud
    onto a plane, while saving height and point density information for each voxel.
    """

    # Class Constants
    VOXEL_EMPTY = -1
    VOXEL_FILLED = 0

    def __init__(self):

        # Quantization size of the voxel grid
        self.voxel_size = 0.0

        # Voxels at the most negative/positive xyz
        self.min_voxel_coord = np.array([])
        self.max_voxel_coord = np.array([])

        # Size of the voxel grid along each axis
        self.num_divisions = np.array([0, 0, 0])

        # Points in sorted order, to match the order of the voxels
        self.points = []

        # Indices of filled voxels
        self.voxel_indices = []

        # Max point height in projected voxel
        self.heights = []

        # Number of points corresponding to projected voxel
        self.num_pts_in_voxel = []

        # Full occupancy grid, VOXEL_EMPTY or VOXEL_FILLED
        self.leaf_layout_2d = []

    def voxelize_2d(self, pts, voxel_size, extents=None,
                    ground_plane=None, create_leaf_layout=True):
        """Voxelizes the point cloud into a 2D voxel grid by
        projecting it down into a flat plane, and stores the maximum
        point height, and number of points corresponding to the voxel

        :param pts: Point cloud as N x [x, y, z] 在相机坐标系下可以投影到图片里，且在地面上方[plan_offset_dist,offset_dist]的点云
        :param voxel_size: Quantization size for the grid
        :param extents: Optional, specifies the full extents of the point cloud.
                        Used for creating same sized voxel grids.
        :param ground_plane: Plane coefficients (a, b, c, d), xz plane used if
                             not specified
        :param create_leaf_layout: Set this to False to create an empty
                                   leaf_layout, which will save computation
                                   time.
        """
        # Check if points are 3D, otherwise early exit
        if pts.shape[1] != 3:
            raise ValueError("Points have the wrong shape: {}".format(
                pts.shape))

        self.voxel_size = voxel_size

        # Discretize voxel coordinates to given quantization size 
        # 将点云坐标离散化为体素voxel，相当于将空间划分为voxel_size大小的网格，将在网格里的点云坐标统一
        # 注意是int型数据，后面会进行去重，这些重复的就是因为将floor变成int，进行量化
        discrete_pts = np.floor(pts / voxel_size).astype(np.int32)

        # Use Lex Sort, sort by x, then z, then y (返回排序后的索引
        x_col = discrete_pts[:, 0]
        y_col = discrete_pts[:, 1]
        z_col = discrete_pts[:, 2]
        sorted_order = np.lexsort((y_col, z_col, x_col))#从x_col从小到大排序，若遇到相同的x_col值则比较y_col...

        # Save original points in sorted order
        self.points = pts[sorted_order]#相机坐标系下可以投影到图片里，且在地面上方[plan_offset_dist,offset_dist]从小到大排序的点云

        # Save discrete points in sorted order
        discrete_pts = discrete_pts[sorted_order]

        # Project all points to a 2D plane
        discrete_pts_2d = discrete_pts.copy()
        discrete_pts_2d[:, 1] = 0#俯视图

        # Format the array to c-contiguous array for unique function
        # ascontiguousarray将变量所占内存变成连续的，可加快运算
        # view:在同一块内存中以不同编码方式读取,https://www.geeksforgeeks.org/numpy-ndarray-view-in-python/
        # discrete_pts_2d:int32  
        contiguous_array = np.ascontiguousarray(discrete_pts_2d).view(
            np.dtype((np.void, discrete_pts_2d.dtype.itemsize *
                      discrete_pts_2d.shape[1])))#np.dtype(np.void,12)12位,原一个int32所占位数*shape[1]，为的是下面一步去重

        # The new coordinates are the discretized array with its unique indexes
        _, unique_indices = np.unique(contiguous_array, return_index=True)#去除重复的点云

        # Sort unique indices to preserve order
        unique_indices.sort()

        voxel_coords = discrete_pts_2d[unique_indices]

        # Number of points per voxel, last voxel calculated separately
        num_points_in_voxel = np.diff(unique_indices)
        num_points_in_voxel = np.append(num_points_in_voxel,
                                        discrete_pts_2d.shape[0] -
                                        unique_indices[-1])#每个去重点 重复的个数

        if ground_plane is None:
            # Use first point in voxel as highest point
            height_in_voxel = self.points[unique_indices, 1]
        else:
            # Ground plane provided
            height_in_voxel = geometry_utils.dist_to_plane(
                ground_plane, self.points[unique_indices])#每个voxel距离plane(x0,y0,z0)的距离,见说明.md

        # Set the height and number of points for each voxel
        self.heights = height_in_voxel
        self.num_pts_in_voxel = num_points_in_voxel

        # Find the minimum and maximum voxel coordinates
        if extents is not None:
            # Check provided extents
            extents_transpose = np.array(extents).transpose()
            if extents_transpose.shape != (2, 3):
                raise ValueError("Extents are the wrong shape {}".format(
                    extents.shape))

            # Set voxel grid extents
            self.min_voxel_coord = np.floor(extents_transpose[0] / voxel_size)
            self.max_voxel_coord = \
                np.ceil((extents_transpose[1] / voxel_size) - 1)

            self.min_voxel_coord[1] = 0#因为投影到2D plane里了
            self.max_voxel_coord[1] = 0#因为投影到2D plane里了

            # Check that points are bounded by new extents 如果voxel点云不在范围里，则报错。
            if not (self.min_voxel_coord <= np.amin(voxel_coords,
                                                    axis=0)).all():
                raise ValueError("Extents are smaller than min_voxel_coord")
            if not (self.max_voxel_coord >= np.amax(voxel_coords,
                                                    axis=0)).all():
                raise ValueError("Extents are smaller than max_voxel_coord")

        else:
            # Automatically calculate extents
            self.min_voxel_coord = np.amin(voxel_coords, axis=0)
            self.max_voxel_coord = np.amax(voxel_coords, axis=0)

        # Get the voxel grid dimensions
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord)
                              + 1).astype(np.int32)#划分的网格个数

        # Bring the min voxel to the origin 体素坐标在体素网格系的索引
        self.voxel_indices = (voxel_coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:#创建体素网格系，网格内包含点云则用0表示，不包含点云则用-1表示
            # Create Voxel Object with -1 as empty/occluded, 0 as occupied
            self.leaf_layout_2d = self.VOXEL_EMPTY * \
                np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout_2d[self.voxel_indices[:, 0], 0,
                                self.voxel_indices[:, 2]] = \
                self.VOXEL_FILLED

    def map_to_index(self, map_index):
        """Converts map coordinate values to 1-based discretized grid index
        coordinate. Note: Any values outside the extent of the grid will be
        forced to be the maximum grid coordinate.

        :param map_index: N x 2 points

        :return: N x length(dim) (grid coordinate)
            [] if min_voxel_coord or voxel_size or grid_index or dim is not set
        """
        if self.voxel_size == 0 \
                or len(self.min_voxel_coord) == 0 \
                or len(map_index) == 0:
            return []

        num_divisions_2d = self.num_divisions[[0, 2]]
        min_voxel_coord_2d = self.min_voxel_coord[[0, 2]]

        # Truncate index (same as np.floor for positive values) and clip
        # to valid voxel index range  超出的部分强置为边界部分
        indices = np.int32(map_index / self.voxel_size) - min_voxel_coord_2d#将map_index体素坐标在体素网格系leaf_layout_2d的索引
        indices[:, 0] = np.clip(indices[:, 0], 0, num_divisions_2d[0])
        indices[:, 1] = np.clip(indices[:, 1], 0, num_divisions_2d[1])

        return indices
