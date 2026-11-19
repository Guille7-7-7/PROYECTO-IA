#!/usr/bin/env python3
"""
scan2maqueta — Escanea un terreno con una cámara y lo convierte en una maqueta 3D (GLB).

Pipeline:
  1. Captura   : cámara en vivo (varias fotos -> panorámica), vídeo o imagen.
  2. IA        : Depth Anything V2 estima la altura del terreno (si hay torch+transformers;
                 si no, usa un estimador clásico de reserva).
  3. Semántica : segmenta agua, caminos, vegetación y edificios; detecta cada árbol.
  4. Maqueta   : malla de terreno texturizada + bloque de diorama con estratos +
                 árboles low-poly + edificios con tejado. Exporta .glb
                 con las mismas piezas que la referencia: terreno / bloque / vegetacion / edificios.

Uso:
  python scan2maqueta.py --camera 0                 # cámara en vivo
  python scan2maqueta.py --image foto.jpg           # una foto (ideal: vista cenital / dron)
  python scan2maqueta.py --video recorrido.mp4      # vídeo -> frames -> panorámica
"""
import argparse
import sys
import numpy as np
import cv2
from scipy import ndimage as ndi
import trimesh
from PIL import Image

# --------------------------------------------------------------------------- #
# 1. CAPTURA
# --------------------------------------------------------------------------- #

def stitch(frames):
    if len(frames) == 1:
        return frames[0]
    st = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = st.stitch(frames)
    if status != cv2.Stitcher_OK:
        print(f"[aviso] No se pudo unir las fotos (código {status}); uso la primera.", file=sys.stderr)
        return frames[0]
    return pano


def capture_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"No se puede abrir la cámara {index}")
    frames = []
    print("ESPACIO = capturar foto | ENTER = terminar | ESC = cancelar")
    print("Consejo: sostén la cámara mirando hacia abajo y solapa ~40% entre fotos.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"Fotos: {len(frames)}  [ESPACIO captura, ENTER termina]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("scan2maqueta", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            frames.append(frame)
        elif k in (13, 10):
            break
        elif k == 27:
            frames = []
            break
    cap.release()
    cv2.destroyAllWindows()
    if not frames:
        sys.exit("No se capturó ninguna foto.")
    return stitch(frames)


def capture_video(path, n=12):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        sys.exit(f"No se puede leer el vídeo {path}")
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return stitch(frames)


def load_input(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"No se puede leer {args.image}")
        return img
    if args.video:
        return capture_video(args.video)
    return capture_camera(args.camera if args.camera is not None else 0)


def prepare(img, max_side=1600):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# 2. IA: ESTIMACIÓN DE ALTURA
# --------------------------------------------------------------------------- #

def height_ai(img_bgr, model_id):
    """Depth Anything V2 -> mapa relativo 0..1 (1 = más cerca de la cámara = más alto)."""
    from transformers import pipeline  # noqa
    import torch  # noqa
    pipe = pipeline("depth-estimation", model=model_id)
    out = pipe(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    d = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    d = cv2.resize(d, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    return d


def height_classic(img_bgr):
    """Reserva sin IA: relieve suave a partir de la luminosidad de baja frecuencia."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), max(img_bgr.shape[:2]) / 25)
    return g


def estimate_height(img_bgr, backend, model_id):
    used = "classic"
    if backend in ("auto", "ai"):
        try:
            d = height_ai(img_bgr, model_id)
            used = "ai"
        except Exception as e:  # sin torch / sin red / sin modelo
            if backend == "ai":
                sys.exit(f"Backend IA no disponible: {e}")
            print(f"[aviso] IA de profundidad no disponible ({type(e).__name__}); "
                  f"uso estimador clásico. Instala torch+transformers para mejor relieve.",
                  file=sys.stderr)
            d = height_classic(img_bgr)
    else:
        d = height_classic(img_bgr)
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return d, used


# --------------------------------------------------------------------------- #
# 3. SEMÁNTICA
# --------------------------------------------------------------------------- #

def clean(mask, open_px=2, min_area=60):
    m = mask.astype(np.uint8)
    if open_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)


def segment(img_bgr, spacing_frac):
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.float32) for i in range(3)]
    amin = W * H * 0.00015

    water = clean((h > 90) & (h < 130) & (s > 70) & (v > 50), 2, amin)
    veg = clean((h > 30) & (h < 90) & (s > 40) & (v > 25), 2, amin)
    road = clean((s < 75) & (v > 170) & ~water, 2, amin * 2)

    # --- edificios: tejados rojizos + cajas claras compactas -------------------
    roof = clean((((h < 11) | (h > 170)) & (s > 110) & (v > 70)), 1, amin * 0.5)
    cand = roof.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(road.astype(np.uint8), connectivity=8)
    for i in range(1, n):  # blobs claros compactos y pequeños = casas, no caminos
        bw_, bh_, a_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if a_ < W * H * 0.01 and max(bw_, bh_) / max(1, min(bw_, bh_)) < 3 and a_ / (bw_ * bh_) > 0.8:
            cand |= lab == i
            road &= lab != i
    buildings = []
    cnts, _ = cv2.findContours(cand.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < amin * 0.6 or a > W * H * 0.03:
            continue
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(c)
        if min(bw, bh) < 3 or max(bw, bh) / min(bw, bh) > 3.2 or a / (bw * bh) < 0.75:
            continue
        buildings.append((cx, cy, bw, bh, ang))

    # --- árboles: copas oscuras dentro de la vegetación ------------------------
    r = max(3, int(W * spacing_frac))
    b_, g_, r_ = [img_bgr[..., i].astype(np.float32) for i in range(3)]
    exg = g_ - np.maximum(r_, b_)                      # "verdor": no se deja engañar por las sombras
    score = cv2.GaussianBlur(exg + 0.25 * (255 - v), (0, 0), r * 0.35)
    local = cv2.GaussianBlur(score, (0, 0), r * 3)
    peaks = (ndi.maximum_filter(score, size=2 * r + 1) == score) & (score - local > 5) & veg
    ys, xs = np.nonzero(peaks)
    trees = np.stack([xs, ys], 1) if len(xs) else np.zeros((0, 2), int)
    return dict(water=water, veg=veg, road=road, buildings=buildings, trees=trees, tree_r=r)


# --------------------------------------------------------------------------- #
# 4. MAQUETA
# --------------------------------------------------------------------------- #

def bilinear(grid, x, y, nx, ny):
    """grid[ny,nx]; x,y en coords de rejilla (float)."""
    return ndi.map_coordinates(grid, [np.clip(y, 0, ny - 1), np.clip(x, 0, nx - 1)], order=1)


def build_ground(d, seg, nx, relief, base, flat_objects=True, ai=True):
    H, W = d.shape
    ny = int(round(nx * H / W))
    dm = d.copy()
    # DSM -> DTM aproximado: quitar la altura de copas y edificios
    obj = cv2.dilate((seg["veg"] | _bmask(seg, W, H)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if flat_objects:
        low = ndi.minimum_filter(dm, size=max(9, W // 25))
        low = cv2.GaussianBlur(low, (0, 0), W / 60)
        dm = np.where(obj, np.minimum(dm, low), dm)
    dm = cv2.GaussianBlur(dm, (0, 0), W / 200)
    g = cv2.resize(dm, (nx, ny), interpolation=cv2.INTER_AREA)
    y = base + relief * g * (1.0 if ai else 0.35)
    # agua: cauce hundido y plano; caminos: aplanados
    wm = cv2.resize(seg["water"].astype(np.float32), (nx, ny), interpolation=cv2.INTER_AREA)
    wm = cv2.GaussianBlur(wm, (0, 0), 0.8)
    rm = cv2.resize(seg["road"].astype(np.float32), (nx, ny), interpolation=cv2.INTER_AREA)
    rm = cv2.GaussianBlur(rm, (0, 0), 0.8)
    smooth = cv2.GaussianBlur(y, (0, 0), 3)
    y = y * (1 - rm) + smooth * rm
    y = y - np.clip(wm * 3, 0, 1) * (relief * 0.18 + 0.6)
    return y.astype(np.float32), ny


def _bmask(seg, W, H):
    m = np.zeros((H, W), np.uint8)
    for cx, cy, bw, bh, ang in seg["buildings"]:
        cv2.fillPoly(m, [cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)], 1)
    return m.astype(bool)


def terrain_mesh(y, nx, ny, wu, du, img_bgr):
    xs = np.linspace(0, wu, nx)
    zs = np.linspace(0, du, ny)
    X, Z = np.meshgrid(xs, zs)
    verts = np.stack([X, y, Z], -1).reshape(-1, 3)
    u, v = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    uv = np.stack([u, 1 - v], -1).reshape(-1, 2)  # trimesh usa origen abajo y lo voltea al exportar glTF
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3),
                            np.stack([b, c, d_], -1).reshape(-1, 3)])
    m = trimesh.Trimesh(verts, faces, process=False)
    tex = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, roughnessFactor=0.95, metallicFactor=0.0,
                                              name="terreno")
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return m


STRATA = [  # color por banda (de arriba a abajo), estilo diorama
    (96, 128, 58), (121, 85, 52), (150, 110, 70), (176, 140, 96), (134, 106, 80), (104, 88, 74),
]


def block_mesh(y, nx, ny, wu, du, rng):
    """Bloque de diorama: cuatro paredes con estratos de color + tapa inferior."""
    top = y
    ymin = float(top.min()) - 1.2
    levels_rel = np.linspace(1, 0, 5)  # 1 .. 0 sobre [0, ymin]
    V, F, C = [], [], []

    def strip(pts_top, p0, p1):
        # pts_top: (n,3) bordes superiores; extrusión vertical por bandas
        n = len(pts_top)
        ys_edges = []
        for t in pts_top[:, 1]:
            ys_edges.append(np.concatenate([[t, t - 1.2], ymin * levels_rel[:-1] * 0 + ymin * levels_rel[:-1], [0]]))
        ys_edges = np.array(ys_edges)  # (n, nb+1)
        ys_edges[:, 2] = np.minimum(ys_edges[:, 2], ys_edges[:, 1])
        for b in range(ys_edges.shape[1] - 1):
            col = np.array(STRATA[min(b, len(STRATA) - 1)], float)
            jitter = rng.normal(0, 5, (n, 1))
            hi = np.column_stack([pts_top[:, 0], ys_edges[:, b], pts_top[:, 2]])
            lo = np.column_stack([pts_top[:, 0], ys_edges[:, b + 1], pts_top[:, 2]])
            base = sum(len(v) for v in V)
            V.append(np.vstack([hi, lo]))
            C.append(np.clip(np.vstack([col + jitter, col + jitter - 8]), 0, 255))
            i = np.arange(n - 1)
            F.append(np.concatenate([np.stack([base + i, base + n + i, base + i + 1], -1),
                                     np.stack([base + i + 1, base + n + i, base + n + i + 1], -1)]))

    xs = np.linspace(0, wu, nx)
    zs = np.linspace(0, du, ny)
    strip(np.column_stack([xs, top[0, :], np.zeros(nx)]), 0, 0)             # borde z=0
    strip(np.column_stack([xs, top[-1, :], np.full(nx, du)]), 0, 0)         # borde z=du
    strip(np.column_stack([np.zeros(ny), top[:, 0], zs]), 0, 0)             # borde x=0
    strip(np.column_stack([np.full(ny, wu), top[:, -1], zs]), 0, 0)         # borde x=wu
    # tapa inferior
    base = sum(len(v) for v in V)
    V.append(np.array([[0, 0, 0], [wu, 0, 0], [wu, 0, du], [0, 0, du]], float))
    C.append(np.tile([70, 60, 52], (4, 1)).astype(float))
    F.append(np.array([[base, base + 1, base + 2], [base, base + 2, base + 3]]))

    v = np.vstack(V)
    f = np.vstack(F)
    c = np.vstack(C).astype(np.uint8)
    m = trimesh.Trimesh(v, f, process=False)
    # orientar caras hacia fuera
    ctr = np.array([wu / 2, ymin / 2, du / 2])
    cen = m.triangles_center
    flip = np.einsum("ij,ij->i", m.face_normals, cen - ctr) < 0
    f[flip] = f[flip][:, ::-1]
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(vertex_colors=np.column_stack([c, np.full(len(c), 255, np.uint8)]))
    return m


def _tree_template(rng):
    can = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    can.apply_scale([1.0, 0.9, 1.0])
    can.apply_translation([0, 1.55, 0])
    trunk = trimesh.creation.cylinder(radius=0.16, height=1.0, sections=6)
    trunk.apply_translation([0, 0.5, 0])
    return can, trunk


def vegetation_mesh(seg, img_bgr, y, nx, ny, wu, du, rng, max_trees):
    H, W = img_bgr.shape[:2]
    pts = seg["trees"]
    if len(pts) == 0:
        return None
    if len(pts) > max_trees:
        pts = pts[rng.choice(len(pts), max_trees, replace=False)]
    sx, sz = wu / W, du / H
    gx = pts[:, 0] / W * (nx - 1)
    gy = pts[:, 1] / H * (ny - 1)
    gy_h = bilinear(y, gx, gy, nx, ny)
    can, trunk = _tree_template(rng)
    unit = wu / W
    scales = seg["tree_r"] * unit * 0.62 * rng.uniform(0.85, 1.2, len(pts))
    verts, faces, cols = [], [], []
    off = 0
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for i, (p, s) in enumerate(zip(pts, scales)):
        pos = np.array([p[0] * sx, gy_h[i] - 0.2, p[1] * sz])
        pix = rgb[p[1], p[0]].astype(float)
        leaf = np.clip(pix * rng.uniform(0.9, 1.25) + [0, 18, 0], 0, 255)
        leaf = leaf * 0.35 + np.array([105, 165, 72]) * 0.65
        cv_ = np.tile(leaf, (len(can.vertices), 1))
        cv_ *= (0.8 + 0.3 * (can.vertices[:, 1:2] - 0.6) / 2.0)   # sombreado vertical suave
        cv_ += rng.normal(0, 6, cv_.shape)
        ct = np.tile([88, 58, 36], (len(trunk.vertices), 1)).astype(float)
        for m_, c_ in ((trunk, ct), (can, cv_)):
            verts.append(m_.vertices * s + pos)
            faces.append(m_.faces + off)
            cols.append(np.clip(c_, 0, 255))
            off += len(m_.vertices)
    v = np.vstack(verts); f = np.vstack(faces); c = np.vstack(cols).astype(np.uint8)
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(vertex_colors=np.column_stack([c, np.full(len(c), 255, np.uint8)]))
    return m


def buildings_mesh(seg, img_bgr, y, nx, ny, wu, du):
    H, W = img_bgr.shape[:2]
    if not seg["buildings"]:
        return None
    sx, sz = wu / W, du / H
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    parts = []
    for cx, cy, bw, bh, ang in seg["buildings"]:
        w_, d_ = bw * sx, bh * sx
        hgt = max(3.0, min(w_, d_) * 0.55)
        ground = float(bilinear(y, np.array([cx / W * (nx - 1)]), np.array([cy / H * (ny - 1)]), nx, ny)[0])
        body = trimesh.creation.box([w_, hgt, d_])
        body.apply_translation([0, hgt / 2, 0])
        ridge = hgt * 0.45
        roof = trimesh.Trimesh(
            vertices=[[-w_ / 2 - .3, hgt, -d_ / 2 - .3], [w_ / 2 + .3, hgt, -d_ / 2 - .3],
                      [w_ / 2 + .3, hgt, d_ / 2 + .3], [-w_ / 2 - .3, hgt, d_ / 2 + .3],
                      [-w_ / 2 - .3, hgt + ridge, 0], [w_ / 2 + .3, hgt + ridge, 0]],
            faces=[[0, 1, 5], [0, 5, 4], [3, 5, 2], [3, 4, 5], [0, 4, 3], [1, 2, 5], [0, 3, 2], [0, 2, 1]],
            process=False)
        roof_c = np.clip(rgb[int(cy), int(cx)].astype(float) * 0.9, 0, 255).astype(np.uint8)
        wall_c = np.array([232, 224, 204], np.uint8)
        body.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.append(wall_c, 255), (len(body.vertices), 1)))
        roof.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.append(roof_c, 255), (len(roof.vertices), 1)))
        part = trimesh.util.concatenate([body, roof])
        # rotación en planta (eje Y) y colocación
        R = trimesh.transformations.rotation_matrix(-np.deg2rad(ang), [0, 1, 0])
        part.apply_transform(R)
        part.apply_translation([cx * sx, ground - 0.3, cy * sz])
        parts.append(part)
    return trimesh.util.concatenate(parts)


def patch_materials(path):
    """trimesh exporta las mallas con color de vértice como metálicas (oscuras). Les pone el
    mismo material mate que la referencia ('color_vertice', rugosidad 0.9, doble cara)."""
    import json, struct
    d = open(path, "rb").read()
    jl, _ = struct.unpack("<II", d[12:20])
    j = json.loads(d[20:20 + jl])
    rest = d[20 + jl:]
    mats = j.setdefault("materials", [])
    mats.append({"name": "color_vertice", "doubleSided": True,
                 "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "roughnessFactor": 0.9,
                                          "metallicFactor": 0.0}})
    mi = len(mats) - 1
    for mesh in j["meshes"]:
        for p in mesh["primitives"]:
            if "COLOR_0" in p["attributes"]:
                p["material"] = mi
            elif "material" in p:
                m = mats[p["material"]].setdefault("pbrMetallicRoughness", {})
                m.update(roughnessFactor=0.95, metallicFactor=0.0)
    jb = json.dumps(j, separators=(",", ":")).encode()
    jb += b" " * (-len(jb) % 4)
    total = 12 + 8 + len(jb) + len(rest)
    open(path, "wb").write(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + rest)


def make_glb(img, d, seg, args, out):
    rng = np.random.default_rng(args.seed)
    H, W = img.shape[:2]
    wu = args.width
    du = wu * H / W
    nx = args.grid
    y, ny = build_ground(d, seg, nx, args.relief, args.base, ai=getattr(args, 'used_ai', True))
    scene = trimesh.Scene()
    scene.add_geometry(terrain_mesh(y, nx, ny, wu, du, img), geom_name="terreno", node_name="terreno")
    scene.add_geometry(block_mesh(y, nx, ny, wu, du, rng), geom_name="bloque", node_name="bloque")
    veg = vegetation_mesh(seg, img, y, nx, ny, wu, du, rng, args.max_trees)
    if veg is not None:
        scene.add_geometry(veg, geom_name="vegetacion", node_name="vegetacion")
    bld = buildings_mesh(seg, img, y, nx, ny, wu, du)
    if bld is not None:
        scene.add_geometry(bld, geom_name="edificios", node_name="edificios")
    scene.export(out)
    patch_materials(out)
    return scene


def main():
    ap = argparse.ArgumentParser(description="Cámara + IA -> maqueta 3D (GLB)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, help="índice de cámara (por defecto 0)")
    src.add_argument("--image", help="foto del terreno (ideal: vista cenital)")
    src.add_argument("--video", help="vídeo recorriendo el terreno")
    ap.add_argument("-o", "--out", default="maqueta.glb")
    ap.add_argument("--backend", choices=["auto", "ai", "classic"], default="auto")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--width", type=float, default=400, help="ancho de la maqueta (unidades)")
    ap.add_argument("--relief", type=float, default=10, help="desnivel máximo del terreno")
    ap.add_argument("--base", type=float, default=22, help="altura mínima del terreno sobre la base")
    ap.add_argument("--grid", type=int, default=200, help="vértices en X de la malla de terreno")
    ap.add_argument("--tree-spacing", type=float, default=0.012,
                    help="separación mínima entre árboles (fracción del ancho de imagen)")
    ap.add_argument("--max-trees", type=int, default=4000)
    ap.add_argument("--save-debug", help="carpeta donde guardar máscaras y mapa de altura")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    img = prepare(load_input(args))
    d, used = estimate_height(img, args.backend, args.model)
    seg = segment(img, args.tree_spacing)
    print(f"altura: {used} | árboles: {len(seg['trees'])} | edificios: {len(seg['buildings'])} | "
          f"agua: {seg['water'].mean():.1%} | caminos: {seg['road'].mean():.1%}")
    if args.save_debug:
        import os
        os.makedirs(args.save_debug, exist_ok=True)
        cv2.imwrite(f"{args.save_debug}/altura.png", (d * 255).astype(np.uint8))
        vis = img.copy()
        vis[seg["water"]] = (255, 120, 0)
        vis[seg["road"]] = (200, 200, 255)
        for x, yy in seg["trees"]:
            cv2.circle(vis, (int(x), int(yy)), 3, (0, 0, 255), -1)
        for cx, cy, bw, bh, ang in seg["buildings"]:
            cv2.polylines(vis, [cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(f"{args.save_debug}/segmentacion.png", vis)
    args.used_ai = used == 'ai'
    make_glb(img, d, seg, args, args.out)
    print(f"Maqueta guardada en {args.out}")


if __name__ == "__main__":
    main()
