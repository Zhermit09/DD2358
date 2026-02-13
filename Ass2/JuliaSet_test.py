import pytest
import JuliaSet

def get_zs_cs_test_data(desired_width):
    x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
    c_real, c_imag = -0.62772, -.42193

    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
    return zs, cs

def test_sum():
    zs,cs = get_zs_cs_test_data(1000)
    assert sum(JuliaSet.calculate_z_serial_purepython(300, zs, cs)) == 33219980