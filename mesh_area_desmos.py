bl_info = {
    "name": "Mesh Area → Desmos",
    "author": "J",
    "version": (1, 6, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Mesh Area",
    "description": "Measure surface area / edge length (whole mesh or selection, including unapplied "
                   "Subdivision Surface and Mirror) and export the geometry to Desmos 3D / 2D",
    "category": "Mesh",
}

import bpy
import numpy as np
import json
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy_extras.io_utils import ExportHelper

FACE_ATTR = "_meshArea_faceSel"
VERT_ATTR = "_meshArea_vertSel"
DESMOS_LIST_LIMIT = 10000
EPS = 1e-6


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

class MESHAREA_Export(bpy.types.PropertyGroup):
    """One measurement kept for the combined Desmos graph."""
    kind: EnumProperty(items=[("2D", "2D", ""), ("3D", "3D", "")])
    tag: IntProperty()
    title: StringProperty()          # folder title in Desmos
    lines: StringProperty()          # Desmos LaTeX, one expression per line
    unit: StringProperty()
    show: BoolProperty(name="Show", default=True,
                       description="Visible in Desmos (re-copy the graph after changing)")
    bbox: FloatVectorProperty(size=6)   # min xyz, max xyz of plotted coords


class MESHAREA_Settings(bpy.types.PropertyGroup):
    source: EnumProperty(
        name="Geometry",
        items=[
            ("EVALUATED", "With Modifiers", "Measure the mesh as you see it, including unapplied "
             "Subdivision Surface, Mirror and other modifiers"),
            ("BASE", "Base Mesh", "Measure the raw cage, ignoring all modifiers"),
        ],
        default="EVALUATED",
    )
    scope: EnumProperty(
        name="Scope",
        items=[
            ("SELECTED", "Selected", "Only what is selected in Edit Mode (faces, edges or vertices)"),
            ("ALL", "Whole Mesh", "Every face of the object"),
        ],
        default="SELECTED",
    )
    space: EnumProperty(
        name="Space",
        items=[
            ("WORLD", "World", "Include the object's location/rotation/scale"),
            ("LOCAL", "Local", "Raw mesh coordinates, ignoring the object transform"),
        ],
        default="WORLD",
    )
    render_levels: BoolProperty(
        name="Use Render Subdiv Levels",
        description="Temporarily evaluate Subdivision Surface / Multires at their Render level "
                    "instead of the Viewport level (more accurate, slower)",
        default=False,
    )
    mirror_both: BoolProperty(
        name="Include Mirrored Side",
        description="On: measure both halves produced by the Mirror modifier. "
                    "Off: only the half you actually modelled",
        default=True,
    )
    plane: EnumProperty(
        name="2D Plane",
        description="How 3D points are flattened for regular (2D) Desmos",
        items=[
            ("FIT", "Best Fit", "Project onto the plane that best fits the points "
             "(exact area for flat selections)"),
            ("XY", "XY (Top)", ""),
            ("XZ", "XZ (Front)", ""),
            ("YZ", "YZ (Side)", ""),
        ],
        default="FIT",
    )
    units: EnumProperty(
        name="Units",
        description="Units used for the readout and every export",
        items=[
            ("SCENE", "Scene", "Follow Scene Properties > Units (unit system, length unit and unit scale)"),
            ("MM", "mm", "Millimetres"),
            ("CM", "cm", "Centimetres"),
            ("M", "m", "Metres"),
            ("IN", "in", "Inches"),
            ("FT", "ft", "Feet"),
        ],
        default="SCENE",
    )
    tag: IntProperty(
        name="Export #",
        description="Number added to every Desmos variable name (U_{1}, A_{loop1}…) so several "
                    "exports can sit on the same graph without clashing",
        default=1, min=1, max=999,
    )
    auto_tag: BoolProperty(
        name="Auto-number",
        description="Increase Export # after every Desmos copy/save",
        default=True,
    )
    centre: BoolProperty(
        name="Centre at Origin",
        description="Shift each 2D export so its centre sits at (0,0) – lines shapes up on top of "
                    "each other for comparison",
        default=False,
    )
    exports: CollectionProperty(type=MESHAREA_Export)
    export_index: IntProperty()
    graph_kind: EnumProperty(name="Graph", items=[("2D", "2D Graph", "desmos.com/calculator"),
                                                  ("3D", "3D Graph", "desmos.com/3d")],
                             default="2D")
    decimals: IntProperty(name="Precision", description="Significant digits kept for coordinates "
                          "(scaled to the mesh size, so tiny meshes stay accurate)", default=7, min=3, max=12)

    # Results (read-only display)
    has_result: BoolProperty(default=False)
    area: FloatProperty(default=0.0)
    length: FloatProperty(default=0.0)
    n_faces: IntProperty(default=0)
    n_tris: IntProperty(default=0)
    n_edges: IntProperty(default=0)
    n_verts: IntProperty(default=0)
    note: StringProperty(default="")
    unit_label: StringProperty(default="")
    loop_area: FloatProperty(default=0.0)
    n_loops: IntProperty(default=0)


# --------------------------------------------------------------------------------------
# Core geometry extraction
# --------------------------------------------------------------------------------------

class MeasureError(Exception):
    pass


def _subdiv_mods(obj):
    return [m for m in obj.modifiers if m.type in {"SUBSURF", "MULTIRES"}]


def _mirror_mods(obj):
    return [m for m in obj.modifiers if m.type == "MIRROR" and m.show_viewport]


def _np_get(coll, attr, n, dtype, width=1):
    arr = np.empty(n * width, dtype=dtype)
    if n:
        coll.foreach_get(attr, arr)
    return arr.reshape(n, width) if width > 1 else arr


def _mirror_keep_filters(obj, base_co, base_mask):
    """For each Mirror modifier build a function: local coords (N,3) -> bool keep mask,
    keeping only the side the original (selected) geometry sits on."""
    filters = []
    ref = base_co[base_mask] if base_mask is not None and base_mask.any() else base_co
    if not len(ref):
        return filters
    for m in _mirror_mods(obj):
        if m.mirror_object:
            T = np.array(m.mirror_object.matrix_world.inverted() @ obj.matrix_world)
        else:
            T = np.eye(4)
        ref_m = ref @ T[:3, :3].T + T[:3, 3]
        for axis in range(3):
            if not m.use_axis[axis]:
                continue
            med = float(ref_m[:, axis].mean())
            if abs(med) < EPS:
                continue  # selection sits on the mirror plane – can't tell sides apart
            side = 1.0 if med > 0 else -1.0
            filters.append((T, axis, side))
    return filters


def _apply_filters(filters, pts_local):
    keep = np.ones(len(pts_local), dtype=bool)
    for T, axis, side in filters:
        p = pts_local @ T[:3, :3].T + T[:3, 3]
        keep &= p[:, axis] * side >= -EPS
    return keep


def extract(context, obj, s):
    """Measure the region and return a dict:
        co (N,3)  – every point used, in the chosen space
        tris (M,3), tri_area (M,)  – selected surface, triangulated
        edges (E,2) – selected wire edges (edge/vertex selections) not inside a selected face
        area, length, n_faces, note
    """
    if obj is None or obj.type != "MESH":
        raise MeasureError("Select a mesh object")

    was_edit = obj.mode == "EDIT"
    if was_edit:
        bpy.ops.object.mode_set(mode="OBJECT")   # flushes edit-mode selection into the mesh

    me = obj.data
    added, saved_levels = [], []
    obj_eval = None
    notes = []
    try:
        use_sel = s.scope == "SELECTED"
        base_co = _np_get(me.vertices, "co", len(me.vertices), np.float64, 3)
        base_vsel = None

        # --- tag selection as attributes so it survives Subdivision/Mirror -----------
        if use_sel:
            fsel = _np_get(me.polygons, "select", len(me.polygons), bool)
            base_vsel = _np_get(me.vertices, "select", len(me.vertices), bool)
            if not fsel.any() and not base_vsel.any():
                raise MeasureError("Nothing selected – select something in Edit Mode, "
                                   "or set Scope to Whole Mesh")
            for name, domain, data in ((FACE_ATTR, "FACE", fsel), (VERT_ATTR, "POINT", base_vsel)):
                if name in me.attributes:
                    me.attributes.remove(me.attributes[name])
                a = me.attributes.new(name, "FLOAT", domain)
                a.data.foreach_set("value", data.astype(np.float32))
                added.append(name)
            me.update()

        # --- choose geometry ---------------------------------------------------------
        if s.source == "EVALUATED":
            if s.render_levels:
                for m in _subdiv_mods(obj):
                    saved_levels.append((m, m.levels))
                    m.levels = m.render_levels
            dg = context.evaluated_depsgraph_get()
            dg.update()
            obj_eval = obj.evaluated_get(dg)
            mesh = obj_eval.to_mesh()
        else:
            mesh = me

        nv, nf, ne = len(mesh.vertices), len(mesh.polygons), len(mesh.edges)
        co_local = _np_get(mesh.vertices, "co", nv, np.float64, 3)
        edges_all = _np_get(mesh.edges, "vertices", ne, np.int64, 2)

        mesh.calc_loop_triangles()
        nt = len(mesh.loop_triangles)
        tris = _np_get(mesh.loop_triangles, "vertices", nt, np.int64, 3)
        tpoly = _np_get(mesh.loop_triangles, "polygon_index", nt, np.int64)

        # --- selection masks ---------------------------------------------------------
        if use_sel:
            fa, va = mesh.attributes.get(FACE_ATTR), mesh.attributes.get(VERT_ATTR)
            if fa is None or va is None:
                notes.append("A modifier dropped the selection – measured whole mesh")
                face_mask = np.ones(nf, dtype=bool)
                vmask = np.zeros(nv, dtype=bool)
            else:
                face_mask = _np_get(fa.data, "value", nf, np.float32) > 0.5
                vmask = _np_get(va.data, "value", nv, np.float32) > 0.999
        else:
            face_mask = np.ones(nf, dtype=bool)
            vmask = np.zeros(nv, dtype=bool)

        # --- mirror: optionally keep only the modelled half ---------------------------
        if s.source == "EVALUATED" and not s.mirror_both and _mirror_mods(obj):
            filters = _mirror_keep_filters(obj, base_co, base_vsel)
            if filters:
                if nf:
                    centres = _np_get(mesh.polygons, "center", nf, np.float64, 3)
                    face_mask &= _apply_filters(filters, centres)
                vmask &= _apply_filters(filters, co_local)
            else:
                notes.append("Selection is on the mirror plane – kept both sides")

        tri_mask = face_mask[tpoly] if nt else np.zeros(0, dtype=bool)
        tris = tris[tri_mask]
        n_faces = int(face_mask.sum()) if nf else 0

        # wire edges: both ends selected, not already part of a selected face
        if use_sel and len(edges_all):
            e_sel = vmask[edges_all[:, 0]] & vmask[edges_all[:, 1]]
            if len(tris):
                tri_e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
                key = lambda e: e[:, 0] * (nv + 1) + e[:, 1]
                in_face = np.isin(key(np.sort(edges_all, axis=1)), key(tri_e))
                e_sel &= ~in_face
            wire = edges_all[e_sel]
        else:
            wire = np.zeros((0, 2), dtype=np.int64)

        # --- transform --------------------------------------------------------------
        co = co_local
        if s.space == "WORLD":
            M = np.array(obj.matrix_world, dtype=np.float64)
            co = co_local @ M[:3, :3].T + M[:3, 3]

        # --- compact to only the vertices we use --------------------------------------
        used = np.zeros(nv, dtype=bool)
        used[tris.ravel()] = True
        used[wire.ravel()] = True
        used |= vmask            # loose selected points
        used_idx = np.nonzero(used)[0]
        remap = np.full(nv, -1, dtype=np.int64)
        remap[used_idx] = np.arange(len(used_idx))
        co_c = co[used_idx]
        tris_c = remap[tris]
        wire_c = remap[wire]

        if len(tris_c):
            a, b, c = co_c[tris_c[:, 0]], co_c[tris_c[:, 1]], co_c[tris_c[:, 2]]
            tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        else:
            tri_area = np.zeros(0)
        length = float(np.linalg.norm(co_c[wire_c[:, 0]] - co_c[wire_c[:, 1]], axis=1).sum()) \
            if len(wire_c) else 0.0

        if not len(co_c):
            raise MeasureError("Nothing left to measure (check the mirror setting / selection)")

        return {
            "co": co_c, "tris": tris_c, "tri_area": tri_area, "edges": wire_c,
            "area": float(tri_area.sum()), "length": length, "n_faces": n_faces,
            "note": "; ".join(notes),
        }
    finally:
        if obj_eval is not None:
            obj_eval.to_mesh_clear()
        for m, lvl in saved_levels:
            m.levels = lvl
        for name in added:
            if name in me.attributes:
                me.attributes.remove(me.attributes[name])
        if added:
            me.update()
        if was_edit:
            bpy.ops.object.mode_set(mode="EDIT")


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------

def _num(v, dec):
    return np.format_float_positional(round(float(v), dec) + 0.0, precision=dec, trim="-")


def _eff_dec(co, sig):
    """Decimal places giving `sig` significant digits for the largest coordinate,
    so tiny (mm-scale) and huge meshes keep the same relative precision."""
    m = float(np.abs(co).max()) if len(co) else 1.0
    if m <= 0:
        return sig
    return max(0, sig - 1 - int(np.floor(np.log10(m))))


# Everything is written as Desmos LaTeX: plain text like "P[T_a]" or "L=0.04"
# is mis-parsed when pasted, LaTeX is not.
def _L(values, dec):
    return r"\left[" + ",".join(_num(v, dec) for v in values) + r"\right]"


def _Li(values):
    return r"\left[" + ",".join(str(int(v)) for v in values) + r"\right]"


def _ix(pt, idx):
    return pt + r"\left[" + idx + r"\right]"


def _fn(name, *args):
    return r"\operatorname{" + name + r"}\left(" + ",".join(args) + r"\right)"


def _project_2d(co, plane):
    if plane == "XY":
        return co[:, [0, 1]]
    if plane == "XZ":
        return co[:, [0, 2]]
    if plane == "YZ":
        return co[:, [1, 2]]
    if len(co) < 2:
        return np.zeros((len(co), 2))
    centre = co.mean(axis=0)
    _, _, vt = np.linalg.svd(co - centre, full_matrices=True)
    return (co - centre) @ vt[:2].T


def _nm(letter, sub, tag):
    """Desmos variable name: one letter + alphanumeric subscript, e.g. T_{a2}.
    The export tag keeps several exports apart on one graph."""
    s = f"{sub}{tag}"
    return f"{letter}_{{{s}}}" if s else letter


def _connectivity(lines, pt, r, area_name, length_name, flat, tag):
    """Triangles, edges and *live* area / length formulas evaluated by Desmos itself."""
    Ta, Tb, Tc = _nm("T", "a", tag), _nm("T", "b", tag), _nm("T", "c", tag)
    Ea, Eb = _nm("E", "a", tag), _nm("E", "b", tag)
    Da, Db, Dc = _nm("D", "a", tag), _nm("D", "b", tag), _nm("D", "c", tag)
    if len(r["tris"]):
        t = r["tris"] + 1   # Desmos lists are 1-indexed
        lines += [f"{Ta}={_Li(t[:, 0])}", f"{Tb}={_Li(t[:, 1])}", f"{Tc}={_Li(t[:, 2])}",
                  _fn("polygon" if flat else "triangle", _ix(pt, Ta), _ix(pt, Tb), _ix(pt, Tc)),
                  f"{Da}=" + _fn("distance", _ix(pt, Tb), _ix(pt, Tc)),
                  f"{Db}=" + _fn("distance", _ix(pt, Ta), _ix(pt, Tc)),
                  f"{Dc}=" + _fn("distance", _ix(pt, Ta), _ix(pt, Tb)),
                  # Heron's formula per triangle, summed
                  area_name + r"=\operatorname{total}\left(\frac{1}{4}\sqrt{\left|"
                  rf"\left({Da}+{Db}+{Dc}\right)\left({Db}+{Dc}-{Da}\right)"
                  rf"\left({Da}+{Dc}-{Db}\right)\left({Da}+{Db}-{Dc}\right)\right|}}\right)"]
    if len(r["edges"]):
        e = r["edges"] + 1
        lines += [f"{Ea}={_Li(e[:, 0])}", f"{Eb}={_Li(e[:, 1])}",
                  # 2D calculator has no segment(); a 2-point polygon draws the same line
                  _fn("polygon" if flat else "segment", _ix(pt, Ea), _ix(pt, Eb)),
                  length_name + "=" + _fn("total", _fn("distance", _ix(pt, Ea), _ix(pt, Eb)))]


def desmos_3d(r, sig, tag=""):
    co = r["co"]
    dec = _eff_dec(co, sig)
    Vx, Vy, Vz, P = _nm("V", "x", tag), _nm("V", "y", tag), _nm("V", "z", tag), _nm("P", "", tag)
    lines = [f"{Vx}={_L(co[:, 0], dec)}", f"{Vy}={_L(co[:, 1], dec)}",
             f"{Vz}={_L(co[:, 2], dec)}", rf"{P}=\left({Vx},{Vy},{Vz}\right)"]
    _connectivity(lines, P, r, _nm("A", "", tag), _nm("L", "", tag), flat=False, tag=tag)
    return "\n".join(lines)


def _closed_loops(edges):
    """Order wire edges into closed loops (every vertex used exactly twice).
    Returns a list of 0-based vertex-index cycles."""
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        # collect this connected component
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for n in adj[v]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(comp) < 3 or any(len(adj[v]) != 2 for v in comp):
            continue  # open chain or branching – not a simple loop
        order, prev, cur = [start], None, start
        while True:
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            if nxt == start:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        loops.append(order)
    return loops


def coords_2d(r, plane, centre):
    co2 = _project_2d(r["co"], plane)
    if centre and len(co2):
        co2 = co2 - co2.mean(axis=0)
    return co2


def desmos_2d(r, sig, plane, tag="", centre=False):
    co2 = coords_2d(r, plane, centre)
    dec = _eff_dec(co2, sig)
    U, W, Q = _nm("U", "", tag), _nm("W", "", tag), _nm("Q", "", tag)
    lines = [f"{U}={_L(co2[:, 0], dec)}", f"{W}={_L(co2[:, 1], dec)}", rf"{Q}=\left({U},{W}\right)"]
    # A_p / L_p are measured on the flattened shape
    _connectivity(lines, Q, r, _nm("A", "p", tag), _nm("L", "p", tag), flat=True, tag=tag)
    # Closed edge loops: order the points and give the area they enclose (shoelace formula)
    loops = _closed_loops(r["edges"]) if len(r["edges"]) else []
    names = []
    for k, loop in enumerate(loops, 1):
        k_s = "" if len(loops) == 1 else f"{k}x"   # e.g. O_{2x1} = loop 2 of export 1
        O, R, A = _nm("O", k_s, tag), _nm("R", k_s, tag), _nm("A", "loop" + k_s, tag)
        o = np.array(loop) + 1
        lines += [f"{O}={_Li(o)}", f"{R}={_Li(np.roll(o, -1))}",
                  _fn("polygon", _ix(Q, O)),
                  A + r"=\frac{1}{2}\left|\operatorname{total}\left(" + _ix(U, O) + _ix(W, R)
                  + "-" + _ix(U, R) + _ix(W, O) + r"\right)\right|"]
        names.append(A)
    if len(names) > 1:
        lines.append(_nm("A", "loops", tag) + "=" + "+".join(names))
    if len(r["tris"]):
        lines.append(f"{_nm('A', '3D', tag)}={_num(r['area'], _eff_dec(np.array([r['area']]), sig))}")
    if len(r["edges"]):
        lines.append(f"{_nm('L', '3D', tag)}={_num(r['length'], _eff_dec(np.array([r['length']]), sig))}")
    return "\n".join(lines)


def points_list(r, sig, tag=""):
    co = r["co"]
    dec = _eff_dec(co, sig)
    return _nm("P", "s", tag) + r"=\left(" + ",".join(_L(co[:, i], dec) for i in range(3)) + r"\right)"


def csv_text(r, dec):
    u = r.get("unit", "BU")
    out = ["# Mesh Area export", f"# units,{u}", f"# area ({u}²),{_num(r['area'], dec)}",
           f"# edge length ({u}),{_num(r['length'], dec)}",
           f"# enclosed area of {r.get('n_loops', 0)} loop(s) ({u}²),{_num(r.get('loop_area', 0.0), dec)}",
           f"# faces,{r['n_faces']},triangles,{len(r['tris'])},edges,{len(r['edges'])}",
           "", f"vertex,x ({u}),y ({u}),z ({u})"]
    for i, (x, y, z) in enumerate(r["co"], 1):
        out.append(f"{i},{_num(x, dec)},{_num(y, dec)},{_num(z, dec)}")
    if len(r["tris"]):
        out += ["", f"triangle,v1,v2,v3,area ({u}²)"]
        for i, (tri, ar) in enumerate(zip(r["tris"] + 1, r["tri_area"]), 1):
            out.append(f"{i},{tri[0]},{tri[1]},{tri[2]},{_num(ar, dec)}")
    if len(r["edges"]):
        out += ["", "edge,v1,v2"]
        for i, e in enumerate(r["edges"] + 1, 1):
            out.append(f"{i},{e[0]},{e[1]}")
    return "\n".join(out)


# metres per unit
UNIT_TABLE = {
    "KILOMETERS": (1000.0, "km"), "METERS": (1.0, "m"), "CENTIMETERS": (0.01, "cm"),
    "MILLIMETERS": (0.001, "mm"), "MICROMETERS": (1e-6, "µm"),
    "MILES": (1609.344, "mi"), "FEET": (0.3048, "ft"), "INCHES": (0.0254, "in"),
    "THOU": (0.0000254, "thou"),
}
CHOICE_TO_UNIT = {"MM": "MILLIMETERS", "CM": "CENTIMETERS", "M": "METERS",
                  "IN": "INCHES", "FT": "FEET"}


def resolve_unit(context, s, co_bu):
    """Return (factor, label): multiply Blender units by `factor` to get `label` units."""
    us = context.scene.unit_settings
    to_m = us.scale_length            # 1 Blender unit = scale_length metres
    if s.units != "SCENE":
        unit_m, label = UNIT_TABLE[CHOICE_TO_UNIT[s.units]]
        return to_m / unit_m, label
    if us.system == "NONE":
        return 1.0, "BU"
    key = us.length_unit
    if key == "ADAPTIVE":
        size_m = float(np.ptp(co_bu, axis=0).max()) * to_m if len(co_bu) else 1.0
        if us.system == "IMPERIAL":
            key = "INCHES" if size_m < 0.3048 else "FEET"
        else:
            key = "MILLIMETERS" if size_m < 0.1 else "CENTIMETERS" if size_m < 1 else "METERS"
    unit_m, label = UNIT_TABLE.get(key, (1.0, "m"))
    return to_m / unit_m, label


def apply_units(context, s, r):
    f, label = resolve_unit(context, s, r["co"])
    r["co"] = r["co"] * f
    r["tri_area"] = r["tri_area"] * f * f
    r["area"] *= f * f
    r["length"] *= f
    r["unit"] = label
    # area enclosed by closed edge loops (vector area: exact for flat outlines,
    # the largest projected area for slightly curved ones)
    loops = _closed_loops(r["edges"]) if len(r["edges"]) else []
    total = 0.0
    for lp in loops:
        p = r["co"][lp]
        total += 0.5 * float(np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))
    r["loop_area"], r["n_loops"] = total, len(loops)
    return r


def _units_note(r, tag=""):
    u = r["unit"]
    head = f"Export {tag}: {r.get('name', '')} – " if tag else ""
    return f'"{head}units {u}  (lengths in {u}, areas in {u}²; names end in {tag or "nothing"})'


def make_text(r, s, fmt):
    tag = str(s.tag)
    if fmt == "DESMOS3D":
        return _units_note(r, tag) + "\n" + desmos_3d(r, s.decimals, tag)
    if fmt == "DESMOS2D":
        return _units_note(r, tag) + "\n" + desmos_2d(r, s.decimals, s.plane, tag, s.centre)
    if fmt == "POINTS":
        return _units_note(r, tag) + "\n" + points_list(r, s.decimals, tag)
    return csv_text(r, _eff_dec(r["co"], s.decimals))


def store_result(s, r):
    s.has_result = True
    s.area = r["area"]
    s.length = r["length"]
    s.n_faces = r["n_faces"]
    s.n_tris = len(r["tris"])
    s.n_edges = len(r["edges"])
    s.n_verts = len(r["co"])
    s.note = r["note"]
    s.unit_label = r["unit"]
    s.loop_area = r.get("loop_area", 0.0)
    s.n_loops = r.get("n_loops", 0)


def size_warning(r, fmt):
    n = max(len(r["co"]), len(r["tris"]), len(r["edges"]))
    if fmt in {"DESMOS3D", "DESMOS2D", "POINTS"} and n > DESMOS_LIST_LIMIT:
        return (f"{n} items exceeds Desmos' {DESMOS_LIST_LIMIT} list limit – "
                "select less or lower the subdivision level")
    return None


FORMATS = [
    ("DESMOS3D", "Desmos 3D", "Points, filled triangles and edges for desmos.com/3d"),
    ("DESMOS2D", "Desmos 2D", "Flattened polygons and edges for desmos.com/calculator"),
    ("POINTS", "Points Only", "Just the coordinates as a Desmos list of 3D points"),
    ("CSV", "CSV", "Spreadsheet-friendly vertex / triangle / edge tables"),
]


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------

class _MeshPoll:
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH"


def _run(op, context):
    s = context.scene.mesh_area
    try:
        r = extract(context, context.active_object, s)
    except MeasureError as e:
        op.report({"WARNING"}, str(e))
        return None
    apply_units(context, s, r)
    r["name"] = context.active_object.name
    store_result(s, r)
    if r["note"]:
        op.report({"WARNING"}, r["note"])
    return r


class MESHAREA_OT_measure(_MeshPoll, bpy.types.Operator):
    bl_idname = "mesh_area.measure"
    bl_label = "Measure"
    bl_description = "Calculate area (selected faces) and length (selected edges)"

    def execute(self, context):
        return {"FINISHED"} if _run(self, context) else {"CANCELLED"}


class MESHAREA_OT_copy(_MeshPoll, bpy.types.Operator):
    bl_idname = "mesh_area.copy"
    bl_label = "Copy"
    bl_description = "Measure and copy the result to the clipboard, ready to paste into Desmos"

    fmt: EnumProperty(items=FORMATS)

    def execute(self, context):
        r = _run(self, context)
        if r is None:
            return {"CANCELLED"}
        s = context.scene.mesh_area
        context.window_manager.clipboard = make_text(r, s, self.fmt)
        used = s.tag
        if self.fmt != "CSV" and s.auto_tag:
            s.tag += 1
        warn = size_warning(r, self.fmt)
        if warn:
            self.report({"WARNING"}, warn)
        elif not r["note"]:
            label = dict((k, n) for k, n, _ in FORMATS)[self.fmt]
            self.report({"INFO"}, f"Copied {label} #{used} in {r['unit']} – paste into an empty Desmos expression")
        return {"FINISHED"}


class MESHAREA_OT_save(_MeshPoll, bpy.types.Operator, ExportHelper):
    bl_idname = "mesh_area.save"
    bl_label = "Save Export"
    bl_description = "Measure and save the result to a text/CSV file"

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt;*.csv", options={"HIDDEN"})
    fmt: EnumProperty(name="Format", items=FORMATS, default="CSV")

    def invoke(self, context, event):
        if self.fmt == "CSV":
            self.filename_ext = ".csv"
        name = context.active_object.name if context.active_object else "mesh"
        self.filepath = bpy.path.ensure_ext(bpy.path.clean_name(name + "_area"), self.filename_ext)
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        r = _run(self, context)
        if r is None:
            return {"CANCELLED"}
        s = context.scene.mesh_area
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(make_text(r, s, self.fmt))
        if self.fmt != "CSV" and s.auto_tag:
            s.tag += 1
        self.report({"INFO"}, f"Saved {self.filepath}")
        return {"FINISHED"}



# --------------------------------------------------------------------------------------
# Combined Desmos graph (one folder per export)
# --------------------------------------------------------------------------------------
# Pasting a Desmos graph-state JSON into an empty expression row loads the whole graph,
# folders included. It replaces what is on the graph, so we always copy every export.

PALETTE = ["#c74440", "#2d70b3", "#388c46", "#6042a6", "#fa7e19", "#000000"]


def _style(latex):
    """Desmos display settings for a generated line."""
    if latex.startswith(r"\operatorname{polygon}"):
        if "O_{" in latex:                       # enclosed loop
            return {"fillOpacity": "0.3", "lineWidth": "2.5"}
        if "T_{a" in latex:                      # flattened faces
            return {"fillOpacity": "0.3", "lineOpacity": "0.15", "lineWidth": "1"}
        return {"lineWidth": "2"}                # edges
    if latex.startswith(r"\operatorname{triangle}"):
        return {"fillOpacity": "0.6", "lineOpacity": "0.2", "lineWidth": "1"}
    if latex.startswith(r"\operatorname{segment}"):
        return {"lineWidth": "2"}
    if latex.startswith(("Q_{", "P_{")) and r"=\left(" in latex:
        return {"hidden": True}                  # raw point cloud – noisy, hidden by default
    return {}


def build_graph_state(exports, kind):
    items = [e for e in exports if e.kind == kind]
    expr = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i, e in enumerate(items):
        fid = f"f{e.tag}"
        colour = PALETTE[i % len(PALETTE)]
        expr.append({"type": "folder", "id": fid, "title": e.title, "collapsed": True,
                     "hidden": not e.show})
        for j, line in enumerate(e.lines.split("\n")):
            eid = f"e{e.tag}_{j}"
            if line.startswith('"'):
                expr.append({"type": "text", "id": eid, "folderId": fid, "text": line[1:]})
                continue
            d = {"type": "expression", "id": eid, "folderId": fid, "latex": line, "color": colour}
            d.update(_style(line))
            expr.append(d)
        if e.show:
            lo = np.minimum(lo, e.bbox[:3])
            hi = np.maximum(hi, e.bbox[3:])
    if not np.isfinite(lo).all():
        lo, hi = np.full(3, -10.0), np.full(3, 10.0)
    centre, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1e-9) * 1.25
    if kind == "2D":
        graph = {"viewport": {"xmin": centre[0] - half, "xmax": centre[0] + half,
                              "ymin": centre[1] - half, "ymax": centre[1] + half}}
    else:
        graph = {"viewport": {"xmin": centre[0] - half, "xmax": centre[0] + half,
                              "ymin": centre[1] - half, "ymax": centre[1] + half,
                              "zmin": centre[2] - half, "zmax": centre[2] + half},
                 "product": "graphing-3d", "threeDMode": True}
    return json.dumps({"version": 11, "graph": graph, "expressions": {"list": expr}},
                      ensure_ascii=False)


def _summary(r, tag):
    u = r["unit"]
    parts = [f"{tag}. {r.get('name', '')}"]
    if len(r["tris"]):
        parts.append(f"area {r['area']:.5g} {u}²")
    if r.get("n_loops"):
        parts.append(f"enclosed {r['loop_area']:.5g} {u}²")
    if len(r["edges"]):
        parts.append(f"length {r['length']:.5g} {u}")
    return "  ·  ".join(parts)


def _copy_graph(context, kind):
    s = context.scene.mesh_area
    n = sum(1 for e in s.exports if e.kind == kind)
    context.window_manager.clipboard = build_graph_state(s.exports, kind)
    return n


class MESHAREA_OT_add_export(_MeshPoll, bpy.types.Operator):
    bl_idname = "mesh_area.add_export"
    bl_label = "Add to Desmos Graph"
    bl_description = ("Measure the selection, add it to the comparison list as its own folder, and copy "
                      "the whole graph – paste into an EMPTY Desmos graph")

    kind: EnumProperty(items=[("2D", "2D", ""), ("3D", "3D", "")])

    def execute(self, context):
        s = context.scene.mesh_area
        r = _run(self, context)
        if r is None:
            return {"CANCELLED"}
        same = [e for e in s.exports if e.kind == self.kind]
        tag = max((e.tag for e in same), default=0) + 1
        if self.kind == "2D":
            lines = desmos_2d(r, s.decimals, s.plane, str(tag), s.centre)
            c = coords_2d(r, s.plane, s.centre)
            c = np.hstack([c, np.zeros((len(c), 1))])
        else:
            lines = desmos_3d(r, s.decimals, str(tag))
            c = r["co"]
        e = s.exports.add()
        e.kind, e.tag, e.unit = self.kind, tag, r["unit"]
        e.title = _summary(r, tag)
        e.lines = f'"{r.get("name", "")} – units {r["unit"]}\n' + lines
        e.bbox = list(c.min(axis=0)) + list(c.max(axis=0)) if len(c) else [0] * 6
        s.export_index = len(s.exports) - 1
        s.graph_kind = self.kind
        n = _copy_graph(context, self.kind)
        units = {x.unit for x in s.exports if x.kind == self.kind}
        if len(units) > 1:
            self.report({"WARNING"}, f"Exports use different units ({', '.join(sorted(units))}) – "
                                     "clear and re-add to compare fairly")
        else:
            self.report({"INFO"}, f"Copied {self.kind} graph with {n} export(s) – paste into an "
                                  "empty Desmos expression (it replaces the graph)")
        return {"FINISHED"}


class MESHAREA_OT_copy_graph(bpy.types.Operator):
    bl_idname = "mesh_area.copy_graph"
    bl_label = "Copy Graph"
    bl_description = "Copy every export in the list as one Desmos graph (one folder each)"

    kind: EnumProperty(items=[("2D", "2D", ""), ("3D", "3D", "")])

    def execute(self, context):
        n = _copy_graph(context, self.kind)
        self.report({"INFO"}, f"Copied {self.kind} graph with {n} export(s) – paste into an empty "
                              "Desmos expression")
        return {"FINISHED"}


class MESHAREA_OT_remove_export(bpy.types.Operator):
    bl_idname = "mesh_area.remove_export"
    bl_label = "Remove Export"
    bl_description = "Remove the selected export from the list"

    def execute(self, context):
        s = context.scene.mesh_area
        if 0 <= s.export_index < len(s.exports):
            s.exports.remove(s.export_index)
            s.export_index = min(s.export_index, len(s.exports) - 1)
        return {"FINISHED"}


class MESHAREA_OT_clear_exports(bpy.types.Operator):
    bl_idname = "mesh_area.clear_exports"
    bl_label = "Clear List"
    bl_description = "Remove every export of this graph type from the list"

    kind: EnumProperty(items=[("2D", "2D", ""), ("3D", "3D", "")])

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        s = context.scene.mesh_area
        for i in reversed(range(len(s.exports))):
            if s.exports[i].kind == self.kind:
                s.exports.remove(i)
        s.export_index = max(0, min(s.export_index, len(s.exports) - 1))
        return {"FINISHED"}


class MESHAREA_UL_exports(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "show", text="", emboss=False,
                 icon="HIDE_OFF" if item.show else "HIDE_ON")
        row.label(text=item.title)

    def filter_items(self, context, data, propname):
        kind = context.scene.mesh_area.graph_kind
        items = getattr(data, propname)
        flags = [self.bitflag_filter_item if it.kind == kind else 0 for it in items]
        return flags, []


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

def _unit_string(s, value, category):
    u = s.unit_label or "BU"
    return f"{value:.6g} {u}²" if category == "AREA" else f"{value:.6g} {u}"


class MESHAREA_PT_panel(bpy.types.Panel):
    bl_label = "Mesh Area  v1.6"
    bl_idname = "MESHAREA_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Area"

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH"

    def draw(self, context):
        s = context.scene.mesh_area
        obj = context.active_object
        L = self.layout

        col = L.column(align=True)
        col.label(text="Measure")
        col.row(align=True).prop(s, "scope", expand=True)
        col.row(align=True).prop(s, "source", expand=True)
        col.row(align=True).prop(s, "space", expand=True)
        col.prop(s, "units")
        if s.source == "EVALUATED":
            if _subdiv_mods(obj):
                col.prop(s, "render_levels")
            if _mirror_mods(obj):
                col.prop(s, "mirror_both")

        L.operator("mesh_area.measure", icon="DRIVER_DISTANCE")

        if s.has_result:
            box = L.box()
            if s.n_tris or not s.n_edges:
                row = box.row()
                row.scale_y = 1.3
                row.label(text="Area: " + _unit_string(s, s.area, "AREA"), icon="MOD_SUBSURF")
            if s.n_edges:
                row = box.row()
                row.scale_y = 1.3
                row.label(text="Length: " + _unit_string(s, s.length, "LENGTH"),
                          icon="EDGESEL")
            if s.n_loops:
                row = box.row()
                row.scale_y = 1.3
                label = "Enclosed: " if s.n_loops == 1 else f"Enclosed ({s.n_loops} loops): "
                row.label(text=label + _unit_string(s, s.loop_area, "AREA"), icon="MESH_CIRCLE")
            g = box.grid_flow(columns=2, align=True)
            g.label(text=f"{s.n_faces} faces")
            g.label(text=f"{s.n_tris} tris")
            g.label(text=f"{s.n_edges} edges")
            g.label(text=f"{s.n_verts} pts")
            if s.note:
                box.label(text=s.note, icon="ERROR")


class MESHAREA_PT_export(bpy.types.Panel):
    bl_label = "Desmos Graph"
    bl_idname = "MESHAREA_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Area"
    bl_parent_id = "MESHAREA_PT_panel"

    def draw(self, context):
        s = context.scene.mesh_area
        L = self.layout

        col = L.column(align=True)
        row = col.row(align=True)
        row.operator("mesh_area.add_export", text="Add 2D", icon="MESH_PLANE").kind = "2D"
        row.prop(s, "plane", text="")
        col.operator("mesh_area.add_export", text="Add 3D", icon="MESH_ICOSPHERE").kind = "3D"
        col.prop(s, "centre")
        if s.plane == "FIT":
            col.label(text="Tip: XY/XZ/YZ keeps shapes aligned", icon="INFO")

        L.row().prop(s, "graph_kind", expand=True)
        L.template_list("MESHAREA_UL_exports", "", s, "exports", s, "export_index", rows=3)
        row = L.row(align=True)
        row.scale_y = 1.3
        row.operator("mesh_area.copy_graph", text="Copy Graph", icon="COPYDOWN").kind = s.graph_kind
        row = L.row(align=True)
        row.operator("mesh_area.remove_export", text="Remove", icon="REMOVE")
        row.operator("mesh_area.clear_exports", text="Clear", icon="TRASH").kind = s.graph_kind
        L.label(text="Paste into an EMPTY Desmos graph", icon="INFO")


class MESHAREA_PT_more(bpy.types.Panel):
    bl_label = "Other Exports"
    bl_idname = "MESHAREA_PT_more"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Area"
    bl_parent_id = "MESHAREA_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.mesh_area
        L = self.layout
        L.label(text="Copy plain lines (append to any graph):")
        col = L.column(align=True)
        row = col.row(align=True)
        row.operator("mesh_area.copy", text="Lines 2D").fmt = "DESMOS2D"
        row.operator("mesh_area.copy", text="Lines 3D").fmt = "DESMOS3D"
        col.operator("mesh_area.copy", text="Selected Points", icon="VERTEXSEL").fmt = "POINTS"
        row = L.row(align=True)
        row.prop(s, "tag")
        row.prop(s, "auto_tag", text="", icon="LINENUMBERS_ON")
        L.prop(s, "decimals")
        row = L.row(align=True)
        row.operator("mesh_area.save", text="Save CSV…", icon="FILE_TICK").fmt = "CSV"
        row.operator("mesh_area.save", text="Save Lines…", icon="EXPORT").fmt = "DESMOS3D"


classes = (MESHAREA_Export, MESHAREA_Settings, MESHAREA_OT_add_export, MESHAREA_OT_copy_graph,
           MESHAREA_OT_remove_export, MESHAREA_OT_clear_exports, MESHAREA_UL_exports, MESHAREA_OT_measure, MESHAREA_OT_copy, MESHAREA_OT_save,
           MESHAREA_PT_panel, MESHAREA_PT_export, MESHAREA_PT_more)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mesh_area = PointerProperty(type=MESHAREA_Settings)


def unregister():
    del bpy.types.Scene.mesh_area
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
