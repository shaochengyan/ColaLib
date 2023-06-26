import open3d


"""
http://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html
"""


def draw_geometries(*args, **kwargs):
    """
    draw_geometries(
        geometry_list, window_name= Open3D , width=1920, height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False
    )
    重载1
        Function to draw a list of geometry.Geometry objects
        geometry_list (List[open3d.geometry.Geometry])   List of geometries to be visualized.
        window_name (str, optional, default='Open3D')   The displayed title of the visualization window.
        width (int, optional, default=1920)   The width of the visualization window.
        height (int, optional, default=1080)   The height of the visualization window.
        left (int, optional, default=50)   The left margin of the visualization window.
        top (int, optional, default=50)   The top margin of the visualization window.
        point_show_normal (bool, optional, default=False)   Visualize point normals if set to true.
        mesh_show_wireframe (bool, optional, default=False)   Visualize mesh wireframe if set to true.
        mesh_show_back_face (bool, optional, default=False)   Visualize also the back face of the mesh triangles.
    重载2
    Parameters
        geometry_list (List[open3d.geometry.Geometry])   List of geometries to be visualized.
        window_name (str, optional, default='Open3D')   The displayed title of the visualization window.
        width (int, optional, default=1920)   The width of the visualization window.
        height (int, optional, default=1080)   The height of the visualization window.
        left (int, optional, default=50)   The left margin of the visualization window.
        top (int, optional, default=50)   The top margin of the visualization window.
        point_show_normal (bool, optional, default=False)   Visualize point normals if set to true.
        mesh_show_wireframe (bool, optional, default=False)   Visualize mesh wireframe if set to true.
        mesh_show_back_face (bool, optional, default=False)   Visualize also the back face of the mesh triangles.
        lookat (numpy.ndarray[numpy.float64[3, 1]])   The lookat vector of the camera.
        up (numpy.ndarray[numpy.float64[3, 1]])   The up vector of the camera.
        front (numpy.ndarray[numpy.float64[3, 1]])   The front vector of the camera.
        zoom (float)   The zoom of the camera.
    """
    return open3d.visualization.draw_geometries(*args, **kwargs)
