import pyvtk


def save(forest, day):
    m, n = forest.shape

    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints(
            dimensions=(m, n, 1),
            origin=(0, 0, 0),
            spacing=(1, 1, 1)
        ),
        pyvtk.PointData(
            pyvtk.Scalars(
                forest.flatten(order='F'),
                name="state"
            )
        )
    )

    filename = f"particles_{day:03d}.vtk"
    vtk_data.tofile(filename)
    print("Saved", filename)
