import numpy as np
import pyvista as pv

COLOR_MAP = {
    1: 'yellow',  
    2: 'red',
    3: 'blue'
}

def render_voxel(volume_3d):
    mask = volume_3d > 0
    pv.plot(mask.astype(int), show_edges=False)

def render_surface(volume_3d):
    pv.start_xvfb()
    plotter = pv.Plotter()

    for cls, color in COLOR_MAP.items():
        mask = (volume_3d == cls)
        if not np.any(mask):
            continue
        grid = pv.wrap(mask.astype(np.uint8))
        surf = grid.contour([0.5])
        plotter.add_mesh(surf, color=color, opacity=0.85)

    plotter.show()

if __name__ == "__main__":
    vol = np.load("predictions_3d/patient_001_3d.npy")
    render_surface(vol)
