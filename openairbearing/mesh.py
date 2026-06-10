"""FEM mesh generation and utilities for bearing geometry.

Provides mesh initialization and manipulation for 2D bearing geometries.
"""

from dataclasses import replace

import numpy as np
from skfem import Mesh2D, MeshQuad1, MeshTri1


class OABMeshTri1(MeshTri1):
    """Custom triangular mesh helpers used by open air bearing geometries."""

    @classmethod
    def init_annulus(
        cls: type,
        inner_diameter: float,
        outer_diameter: float,
        ntheta: int = 24,
        nr: int = 12,
    ) -> Mesh2D:
        """Create a triangular annulus mesh with tagged inner and outer boundaries."""
        if inner_diameter <= 0 or outer_diameter <= 0:
            raise ValueError("Diameters must be positive.")
        if outer_diameter <= inner_diameter:
            raise ValueError("outer_diameter must be greater than inner_diameter.")
        if ntheta < 8:
            raise ValueError("ntheta must be >= 8.")
        if nr < 2:
            raise ValueError("nr must be >= 2.")

        ri = 0.5 * inner_diameter
        ro = 0.5 * outer_diameter

        theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        radii = np.linspace(ri, ro, nr + 1)

        p = np.zeros((2, (nr + 1) * ntheta), dtype=np.float64)
        for j, rj in enumerate(radii):
            base = j * ntheta
            p[0, base : base + ntheta] = rj * np.cos(theta)
            p[1, base : base + ntheta] = rj * np.sin(theta)

        tris = []
        for j in range(nr):
            b0 = j * ntheta
            b1 = (j + 1) * ntheta
            for k in range(ntheta):
                kp = (k + 1) % ntheta
                a = b0 + k
                b = b0 + kp
                c = b1 + k
                d = b1 + kp
                tris.append([a, c, d])
                tris.append([a, d, b])

        t = np.asarray(tris, dtype=np.int32).T
        m = cls(p, t)

        f0, f1 = m.facets
        ring0 = (f0 // ntheta).astype(np.int32)
        ring1 = (f1 // ntheta).astype(np.int32)

        inner_facets = np.where((ring0 == 0) & (ring1 == 0))[0].astype(np.int32)
        outer_facets = np.where((ring0 == nr) & (ring1 == nr))[0].astype(np.int32)

        m = replace(
            m,
            _boundaries={
                "inner": np.sort(inner_facets),
                "outer": np.sort(outer_facets),
            },
        )
        return m


class OABMeshQuad1(MeshQuad1):
    """Custom quadrilateral mesh helpers used by open air bearing geometries."""

    @classmethod
    def init_circle(
        cls: type,
        diameter: float,
        nc: int = 8,
        nr: int = 4,
        ratio: float = 0.7,
    ) -> Mesh2D:
        """Create a quadrilateral O-grid disk mesh with a tagged outer boundary.

        The mesh uses an O-grid topology: a central square block of *nc* x *nc*
        quads surrounded by *nr* radial layers that smoothly transition from the
        square boundary to the circular boundary.

        Parameters
        ----------
        diameter : float
            Circle diameter (must be positive).
        nc : int
            Number of elements per side of the central square block (>= 2).
        nr : int
            Number of radial element layers from central square to circle (>= 1).
        ratio : float
            Ratio of the central square corner radius to the circle radius.
            Controls how much of the domain is covered by the structured core
            (0 < ratio < 1). Default 0.7.
            The side boundaries are smoothly bell-shaped to improve cell
            transition quality near corners.
        """
        if diameter <= 0:
            raise ValueError("Diameter must be positive.")
        if nc < 2:
            raise ValueError("nc must be >= 2.")
        if nr < 1:
            raise ValueError("nr must be >= 1.")
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio must be between 0 and 1 (exclusive).")

        radius = 0.5 * diameter
        # Ratio is defined on the square-corner radius, not axis-aligned half-side.
        half_side = (ratio * radius) / np.sqrt(2.0)

        # ---- Central structured block: (nc+1)^2 nodes ----
        xs = np.linspace(-half_side, half_side, nc + 1)
        ys = np.linspace(-half_side, half_side, nc + 1)
        gx, gy = np.meshgrid(xs, ys)
        center_p = np.array([gx.ravel(), gy.ravel()])

        # Apply a smooth bell-shaped warp field over the full inner block:
        # - zero deformation at corners
        # - maximum at side midpoints
        # - smoothly distributed through interior cells
        # This improves core cell quality and propagates smoothly into wraps.
        gap_to_circle_mid = radius - half_side
        bell_amplitude = 0.35 * gap_to_circle_mid

        if bell_amplitude > 0.0:
            xi = np.clip(center_p[0] / half_side, -1.0, 1.0)
            eta = np.clip(center_p[1] / half_side, -1.0, 1.0)

            # Side-normal displacements for all nodes.
            dx = bell_amplitude * xi * (1.0 - eta**2) ** 2
            dy = bell_amplitude * eta * (1.0 - xi**2) ** 2
            center_p[0] += dx
            center_p[1] += dy

        n_center = center_p.shape[1]

        center_quads = []
        for i in range(nc):
            for j in range(nc):
                n0 = i * (nc + 1) + j
                n1 = n0 + 1
                n2 = n0 + (nc + 2)
                n3 = n0 + (nc + 1)
                center_quads.append([n0, n1, n2, n3])

        # ---- Square perimeter nodes (CCW traversal) ----
        perim = []
        for j in range(nc):
            perim.append(j)
        for i in range(nc):
            perim.append(i * (nc + 1) + nc)
        for j in range(nc, 0, -1):
            perim.append(nc * (nc + 1) + j)
        for i in range(nc, 0, -1):
            perim.append(i * (nc + 1))
        perim = np.asarray(perim, dtype=np.int64)
        n_perim = len(perim)  # 4 * nc

        sq_xy = center_p[:, perim]
        angles = np.arctan2(sq_xy[1], sq_xy[0])
        circ_xy = radius * np.array([np.cos(angles), np.sin(angles)])

        # ---- Outer radial layers (1 … nr) ----
        outer_p = np.zeros((2, n_perim * nr), dtype=np.float64)
        for k in range(1, nr + 1):
            t = k / nr
            start = (k - 1) * n_perim
            outer_p[:, start : start + n_perim] = (1.0 - t) * sq_xy + t * circ_xy

        # ---- Outer ring quads ----
        outer_quads = []
        for r in range(nr):
            for m in range(n_perim):
                mp = (m + 1) % n_perim
                if r == 0:
                    i0, i1 = int(perim[m]), int(perim[mp])
                else:
                    i0 = n_center + (r - 1) * n_perim + m
                    i1 = n_center + (r - 1) * n_perim + mp
                i2 = n_center + r * n_perim + mp
                i3 = n_center + r * n_perim + m
                outer_quads.append([i0, i1, i2, i3])

        # ---- Assemble mesh ----
        p = np.hstack([center_p, outer_p])
        t = np.asarray(center_quads + outer_quads, dtype=np.int64).T
        m = cls(p, t)

        m = replace(m, _boundaries={"outer": m.boundary_facets()})
        return m

    @classmethod
    def init_annulus(
        cls: type,
        inner_diameter: float,
        outer_diameter: float,
        nr: int = 12,
        ntheta: int = 24,
    ) -> Mesh2D:
        """Create a quadrilateral annulus mesh.

        The generated mesh includes tagged inner and outer boundaries.
        """
        if inner_diameter <= 0 or outer_diameter <= 0:
            raise ValueError("Diameters must be positive.")
        if outer_diameter <= inner_diameter:
            raise ValueError("outer_diameter must be greater than inner_diameter.")
        if ntheta < 8:
            raise ValueError("ntheta must be >= 8.")
        if nr < 2:
            raise ValueError("nr must be >= 2.")

        ri = 0.5 * inner_diameter
        ro = 0.5 * outer_diameter

        theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        radii = np.linspace(ri, ro, nr + 1)

        p = np.zeros((2, (nr + 1) * ntheta), dtype=np.float64)
        for j, rj in enumerate(radii):
            base = j * ntheta
            p[0, base : base + ntheta] = rj * np.cos(theta)
            p[1, base : base + ntheta] = rj * np.sin(theta)

        quads = []
        for j in range(nr):
            b0 = j * ntheta
            b1 = (j + 1) * ntheta
            for k in range(ntheta):
                kp = (k + 1) % ntheta
                a = b0 + k
                b = b0 + kp
                c = b1 + k
                d = b1 + kp
                quads.append([a, b, d, c])

        t = np.asarray(quads, dtype=np.int32).T
        m = cls(p, t)

        f0, f1 = m.facets
        ring0 = (f0 // ntheta).astype(np.int32)
        ring1 = (f1 // ntheta).astype(np.int32)

        inner_facets = np.where((ring0 == 0) & (ring1 == 0))[0].astype(np.int32)
        outer_facets = np.where((ring0 == nr) & (ring1 == nr))[0].astype(np.int32)

        m = replace(
            m,
            _boundaries={
                "inner": np.sort(inner_facets),
                "outer": np.sort(outer_facets),
            },
        )
        return m
